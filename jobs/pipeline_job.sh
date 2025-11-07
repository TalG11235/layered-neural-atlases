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

RESULTS_FOLDER=$(grep -oP '"results_folder_name"\s*:\s*"\K[^"]+' "$CONFIG_PATH")
echo "Trainer results folder: $RESULTS_FOLDER"

exp_train_dir=$(find "$RESULTS_FOLDER"/ -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)
if [[ -z "${exp_train_dir:-}" ]]; then
  echo "ERROR: No training directory found under $RESULTS_FOLDER"
  exit 1
fi

exp_dir_name="$(basename "$exp_train_dir")"
EXPERIMENT_ROOT="$EXPERIMENTS_DIR/$exp_dir_name"
TRAINING_RESULTS_DIR="$EXPERIMENT_ROOT/training results"
COMPRESSION_RESULTS_DIR="$EXPERIMENT_ROOT/compression results"
EVAL_DIR_ROOT="$EXPERIMENT_ROOT/eval"

TMP_DIR="$EXPERIMENT_ROOT/_tmp"
RAW_RGB_DIR="$TMP_DIR/raw_textures_rgb"          # << new: raw inputs for RDEIC
COMPRESSED_RGB_DIR="$TMP_DIR/compressed_atlases_rgb"
CHECKPOINT_WORK_DIR="$TMP_DIR/checkpoint"

mkdir -p "$EXPERIMENT_ROOT" \
         "$TRAINING_RESULTS_DIR" "$COMPRESSION_RESULTS_DIR" "$EVAL_DIR_ROOT" \
         "$TMP_DIR" "$RAW_RGB_DIR" "$COMPRESSED_RGB_DIR" "$CHECKPOINT_WORK_DIR"

echo "Moving training results -> $TRAINING_RESULTS_DIR"
shopt -s dotglob nullglob
mv "$exp_train_dir"/* "$TRAINING_RESULTS_DIR"/
shopt -u dotglob nullglob
rm -rf "$exp_train_dir"
rmdir --ignore-fail-on-non-empty "$RESULTS_FOLDER" || true

latest_step_dir=$(find "$TRAINING_RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)
echo "Latest step dir: $latest_step_dir"

# ---------------- Use raw textures directly (no prep / no alpha) -------------
echo "=== Using raw textures from training results (RDEIC input) ==="
TEXTURE1_SRC="$latest_step_dir/texture_orig1.png"
TEXTURE2_SRC="$latest_step_dir/texture_orig2.png"

for f in "$TEXTURE1_SRC" "$TEXTURE2_SRC"; do
  [[ -f "$f" ]] || { echo "ERROR: Missing $f"; exit 1; }
done

# Copy raw inputs to a clean folder with the same names RDEIC will preserve
cp "$TEXTURE1_SRC" "$RAW_RGB_DIR/texture_orig1.png"
cp "$TEXTURE2_SRC" "$RAW_RGB_DIR/texture_orig2.png"

# ----------------------- UNUSED: atlas prep / masks / attach -----------------
# UNUSED: atlas_prep.py, mask RGB expansion, RDEIC mask compression, grayscale, atlas_attach.py

# -------------------------------- RDEIC --------------------------------------
echo "=== Atlases Compression (RDEIC on raw textures) ==="
safe_activate rdeic
pushd "$RDEIC_DIR" >/dev/null

python3 "$RDEIC_DIR/inference_partition.py" \
  --ckpt_sd "$WEIGHT_DIR/v2-1_512-ema-pruned.ckpt" \
  --ckpt_cc "$WEIGHT_DIR/rdeic_2_step2.ckpt" \
  --config "$RDEIC_DIR/configs/model/rdeic.yaml" \
  --input "$RAW_RGB_DIR" \
  --output "$COMPRESSED_RGB_DIR" \
  --steps 2 \
  --guidance_scale 1 \
  --device cuda

popd >/dev/null

# Put compressed outputs where the rest of the pipeline expects them
cp "$COMPRESSED_RGB_DIR/texture_orig1.png" "$COMPRESSION_RESULTS_DIR/texture_orig1.png"
cp "$COMPRESSED_RGB_DIR/texture_orig2.png" "$COMPRESSION_RESULTS_DIR/texture_orig2.png"
cp "$CONFIG_PATH" "$COMPRESSION_RESULTS_DIR/"

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

# ---------------------------- Packaging ---------------------------
echo "=== Packaging ==="
python "$PROJECT_SRC_DIR/package_experiment.py" "$EXPERIMENT_ROOT"

# --------------------------- Evaluation -------------------------------
echo "=== Evaluation ==="
mkdir -p "$EVAL_DIR_ROOT"
python "$PROJECT_SRC_DIR/evaluate_experiment.py" --experiment_dir "$EXPERIMENT_ROOT"

echo "Done. Experiment root: $EXPERIMENT_ROOT"
