#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --nodelist=lambda3
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

echo "Job started on $(hostname)"
echo "CWD: $(pwd)"

#### Run from Layered neural atlases project root ####

# --- Safe Conda activation (prevents unbound var errors from activate hooks) ---
safe_activate() {
  local env_name="$1"
  set +u
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "$env_name"
  set -u
}

# ------------------------------- Constants -----------------------------------
PROJECT_DIR="$(pwd)"
PROJECT_SRC_DIR="$PROJECT_DIR/src"
EXPERIMENTS_DIR="$PROJECT_DIR/experiments"
RDEIC_DIR="$PROJECT_DIR/thirdparty/RDEIC"
WEIGHT_DIR="$RDEIC_DIR/weight"
CONFIG_PATH="$1"

mkdir -p "$EXPERIMENTS_DIR"

# ------------------------------- Training ------------------------------------
echo "=== Training ==="
safe_activate neural_atlases

python "$PROJECT_SRC_DIR/train.py" "$CONFIG_PATH"

# Grab trainer output folder name from config
RESULTS_FOLDER=$(grep -oP '"results_folder_name"\s*:\s*"\K[^"]+' "$CONFIG_PATH")
echo "Trainer results folder: $RESULTS_FOLDER"

# Most recent run produced by the trainer
exp_train_dir=$(find "$RESULTS_FOLDER"/ -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)
if [[ -z "${exp_train_dir:-}" ]]; then
  echo "ERROR: No training directory found under $RESULTS_FOLDER"
  exit 1
fi

exp_dir_name="$(basename "$exp_train_dir")"            # EXP_NAME
EXPERIMENT_ROOT="$EXPERIMENTS_DIR/$exp_dir_name"
TRAINING_RESULTS_DIR="$EXPERIMENT_ROOT/training results"
COMPRESSION_RESULTS_DIR="$EXPERIMENT_ROOT/compression results"
EVAL_DIR_ROOT="$EXPERIMENT_ROOT/eval"

TMP_DIR="$EXPERIMENT_ROOT/_tmp"                        # << all temps here
PREP_RGB_DIR="$TMP_DIR/prep_atlases_rgb"
PREP_MASK_DIR="$TMP_DIR/prep_atlases_masks"
COMPRESSED_RGB_DIR="$TMP_DIR/compressed_atlases_rgb"
FINAL_RGBA_DIR="$TMP_DIR/compressed_atlases_rgba"
CHECKPOINT_WORK_DIR="$TMP_DIR/checkpoint"

mkdir -p "$EXPERIMENT_ROOT" \
         "$TRAINING_RESULTS_DIR" "$COMPRESSION_RESULTS_DIR" "$EVAL_DIR_ROOT" \
         "$PREP_RGB_DIR" "$PREP_MASK_DIR" "$COMPRESSED_RGB_DIR" "$FINAL_RGBA_DIR" "$CHECKPOINT_WORK_DIR"

# Move training results into the experiment
echo "Moving training results -> $TRAINING_RESULTS_DIR"
shopt -s dotglob nullglob
mv "$exp_train_dir"/* "$TRAINING_RESULTS_DIR"/
shopt -u dotglob nullglob
rm -rf "$exp_train_dir"
rmdir --ignore-fail-on-non-empty "$RESULTS_FOLDER" || true

# Latest numeric step dir inside training results (e.g., 200000)
latest_step_dir=$(find "$TRAINING_RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)
echo "Latest step dir: $latest_step_dir"

# ----------------------- Atlases Pre-processing (alpha/bleed) ----------------
echo "=== [Alpha] Preparing atlases for compression ==="
safe_activate neural_atlases

python "$PROJECT_SRC_DIR/atlas_prep.py" \
  --eval_dir "$latest_step_dir" \
  --out_rgb_dir "$PREP_RGB_DIR" \
  --out_mask_dir "$PREP_MASK_DIR" \
  --names texture_orig1.png \
  --feather_sigma 0.6 \
  --bleed_px 12 \
  --inner_offset_px 2

python "$PROJECT_SRC_DIR/atlas_prep.py" \
  --eval_dir "$latest_step_dir" \
  --out_rgb_dir "$PREP_RGB_DIR" \
  --out_mask_dir "$PREP_MASK_DIR" \
  --names texture_orig2.png \
  --bleed_px 20 \
  --inner_offset_px 0 \
  --feather_sigma 0.6

# ------------------------------- Compression ---------------------------------
echo "=== Atlases Compression (RDEIC) ==="
safe_activate rdeic
pushd "$RDEIC_DIR" >/dev/null

python3 "$RDEIC_DIR/inference_partition.py" \
  --ckpt_sd "$WEIGHT_DIR/v2-1_512-ema-pruned.ckpt" \
  --ckpt_cc "$WEIGHT_DIR/rdeic_2_step2.ckpt" \
  --config configs/model/rdeic.yaml \
  --input "$PREP_RGB_DIR" \
  --output "$COMPRESSED_RGB_DIR" \
  --steps 2 \
  --guidance_scale 1 \
  --device cuda

popd >/dev/null

# ---------------------------- Re-attach Alpha --------------------------------
echo "=== Atlases Alpha Layers (attach) ==="
safe_activate neural_atlases

python "$PROJECT_SRC_DIR/atlas_attach.py" \
  --comp_rgb_dir "$COMPRESSED_RGB_DIR" \
  --mask_dir "$PREP_MASK_DIR" \
  --out_rgba_dir "$FINAL_RGBA_DIR" \
  --names texture_orig1.png

python "$PROJECT_SRC_DIR/atlas_attach.py" \
  --comp_rgb_dir "$COMPRESSED_RGB_DIR" \
  --mask_dir "$PREP_MASK_DIR" \
  --out_rgba_dir "$FINAL_RGBA_DIR" \
  --names texture_orig2.png \
  --alpha-one

echo "[Alpha] Final premultiplied RGBA atlases written to: $FINAL_RGBA_DIR"

# -------------------------- Prepare Compression Output -----------------------
echo "=== Prepare Compression Output ==="
cp "$CONFIG_PATH" \
   "$FINAL_RGBA_DIR/texture_orig1.png" \
   "$FINAL_RGBA_DIR/texture_orig2.png" \
   "$COMPRESSION_RESULTS_DIR/"

# --------------------------- Checkpoint Reduction ----------------------------
echo "=== Checkpoint Reduction ==="
safe_activate neural_atlases
export TRAINING_RESULTS_DIR CHECKPOINT_WORK_DIR

python <<'PY'
import os, torch
training_root = os.environ["TRAINING_RESULTS_DIR"]
ckpt_path = os.path.join(training_root, "checkpoint")
print(f"Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location="cpu")
checkpoint.pop("F_atlas_state_dict", None)
checkpoint.pop("optimizer_all_state_dict", None)
out_dir = os.environ["CHECKPOINT_WORK_DIR"]
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "checkpoint")
torch.save(checkpoint, out_path)
print(f"Saved reduced checkpoint -> {out_path}")
PY

# ------------------------- Checkpoint Quantization ---------------------------
echo "=== Checkpoint Quantization ==="
safe_activate neural_atlases
python "$PROJECT_SRC_DIR/quantize_checkpoint.py" \
       "$CONFIG_PATH" \
       "$CHECKPOINT_WORK_DIR/checkpoint" \
       "$COMPRESSION_RESULTS_DIR/checkpoint"

# ---------------------------- Video Reconstruction ---------------------------
echo "=== Video Reconstruction ==="
video_name=$(basename "$(dirname -- "$CONFIG_PATH")")

python "$PROJECT_SRC_DIR/only_edit.py" \
  --trained_model_folder="$COMPRESSION_RESULTS_DIR/" \
  --video_name="$video_name" \
  --output_folder="$COMPRESSION_RESULTS_DIR/reconstruction/" \
  --edit_foreground_path="$COMPRESSION_RESULTS_DIR/texture_orig1.png" \
  --edit_background_path="$COMPRESSION_RESULTS_DIR/texture_orig2.png"

# ------------------------------- Create /eval -------------------------------
mkdir -p "$EVAL_DIR_ROOT"

echo "Done. Experiment root: $EXPERIMENT_ROOT"
