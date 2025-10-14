#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --nodelist=lambda3
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

#SBATCH --output=logs/%x-%j.out

echo "Job started on $(hostname)"

#### This script is meant to be used from the Layered neural atlases project directory ####

# general variables for the pipline
PROJECT_DIR="$(pwd)"
PROJECT_SRC_DIR="$PROJECT_DIR/src"
EXPERIMENTS_DIR="$PROJECT_DIR/experiments"
EXPERIMENTS_RESULTS_DIR="$EXPERIMENTS_DIR/results"
ATLASES_DIR="$EXPERIMENTS_DIR/atlases"
RDEIC_DIR="$(pwd)/thirdparty/RDEIC"
WEIGHT_DIR="$RDEIC_DIR/weight"
CONFIG_PATH="$1"
ATLAS_PROJECT_DIR="$(pwd)"
RGB_ATLASES_DIR="$EXPERIMENTS_DIR/compressed_atlases"
ALPHA_ATLASES_OUTPUT_DIR="$EXPERIMENTS_DIR/compressed_atlases_alpha"
CHECKPOINT_DIR="$EXPERIMENTS_DIR/checkpoint"
# create directory for experiments if doesn't exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
  mkdir -p "$EXPERIMENTS_DIR"
fi
if [ ! -d "$EXPERIMENTS_RESULTS_DIR" ]; then
  mkdir -p "$EXPERIMENTS_RESULTS_DIR"
fi

##################################### Training ################################################
echo "Training"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural_atlases

python $PROJECT_SRC_DIR/train.py $CONFIG_PATH

# Navigate into the latest directory inside the results folder specified in the config file
RESULTS_FOLDER=$(grep -oP '"results_folder_name"\s*:\s*"\K[^"]+' "$CONFIG_PATH")
echo "Config Path: $CONFIG_PATH"
echo "Using results folder: $RESULTS_FOLDER"

exp_dir=$(find "$RESULTS_FOLDER"/ -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)
mv $exp_dir $EXPERIMENTS_RESULTS_DIR
rm -rf "$RESULTS_FOLDER"

# Find the subdirectory with the highest number (frame index) for later use
latest_experiment="$EXPERIMENTS_RESULTS_DIR/$(basename "$exp_dir")"
latest_experiment_evaluation=$(find "$latest_experiment" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)
exp_dir_name=$(basename "$latest_experiment")

################################ Atlases Pre-proccessing ##########################################
echo "[Alpha] Preparing atlases (bleed) for compression"

EVAL_DIR="$latest_experiment_evaluation"
PREP_RGB_DIR="$EXPERIMENTS_DIR/prep_atlases_rgb"
PREP_MASK_DIR="$EXPERIMENTS_DIR/prep_atlases_masks"
NAMES=("texture_orig1.png" "texture_orig2.png")

mkdir -p "$PREP_RGB_DIR" "$PREP_MASK_DIR"

python "$PROJECT_SRC_DIR/atlas_prep.py" \
  --eval_dir "$EVAL_DIR" \
  --out_rgb_dir "$PREP_RGB_DIR" \
  --out_mask_dir "$PREP_MASK_DIR" \
  --names "${NAMES[@]}" \
  --feather_sigma 0.8 \
  --bleed_px 120 \
  --key_thr 15

################################ Atlases Compression ##########################################
COMPRESSED_RGB_DIR="$EXPERIMENTS_DIR/compressed_atlases_rgb"
mkdir -p "$COMPRESSED_RGB_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdeic

cd $RDEIC_DIR

python3 $RDEIC_DIR/inference_partition.py \
--ckpt_sd $WEIGHT_DIR/v2-1_512-ema-pruned.ckpt \
--ckpt_cc $WEIGHT_DIR/rdeic_2_step2.ckpt \
--config configs/model/rdeic.yaml \
--input $PREP_RGB_DIR \
--output $COMPRESSED_RGB_DIR \
--steps 2 \
--guidance_scale 1 \
--device cuda 

cd $PROJECT_DIR

##################################### Atlases Alpha Layers ################################################
# After decompression, re-attach the original alpha
echo "Atlases Alpha Layers"
source ~/miniconda3/etc/profile.d/conda.sh

conda activate neural_atlases

FINAL_RGBA_DIR="$EXPERIMENTS_DIR/compressed_atlases_rgba"
mkdir -p "$FINAL_RGBA_DIR"

python "$PROJECT_SRC_DIR/atlas_attach.py" \
  --comp_rgb_dir "$COMPRESSED_RGB_DIR" \
  --out_rgba_dir "$FINAL_RGBA_DIR" \
  --alpha 1.0 \
  --names "${NAMES[@]}"

echo "[Alpha] Final premultiplied RGBA atlases written to: $FINAL_RGBA_DIR"

##################################### Prepare Final Output ################################################
echo "Prepare Final Output"

# create output folder
mkdir "$EXPERIMENTS_DIR/outputs/"
final_output_folder="$EXPERIMENTS_DIR/outputs/$exp_dir_name"
# copy everything to one folder
mkdir -p "$final_output_folder"

cp $CONFIG_PATH $FINAL_RGBA_DIR/texture_orig1.png $FINAL_RGBA_DIR/texture_orig2.png $final_output_folder

##################################### Checkpoint Reduction ################################################
echo "Checkpoint Reduction"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural_atlases

mkdir -p "$CHECKPOINT_DIR"

python <<EOF
import torch

checkpoint_path="${latest_experiment%/}/checkpoint"
print(checkpoint_path)
checkpoint = torch.load(checkpoint_path)

# Remove the specified keys
checkpoint.pop("F_atlas_state_dict", None)
checkpoint.pop("optimizer_all_state_dict", None)

# Save the new checkpoint to compressed_results
torch.save(checkpoint, "$CHECKPOINT_DIR/checkpoint")
EOF

##################################### Checkpoint Quiantization ################################################
echo "Checkpoint Quiantization"
# quantization
source ~/miniconda3/etc/profile.d/conda.sh

conda activate neural_atlases

python $PROJECT_SRC_DIR/quantize_checkpoint.py $CONFIG_PATH $CHECKPOINT_DIR/checkpoint $final_output_folder/checkpoint

##################################### Video Reconstruction ################################################
echo "Video Reconstruction"
video_name=$(basename "$(dirname -- "$CONFIG_PATH")")

# recounstruct video with compressed atlases
python $PROJECT_SRC_DIR/only_edit.py --trained_model_folder=$final_output_folder/ \
--video_name="$video_name" \
--output_folder="$final_output_folder/reconstruction/" \
--edit_foreground_path="$final_output_folder"/texture_orig1.png \
--edit_background_path="$final_output_folder"/texture_orig2.png



# clear previous intermidiate directories
rm -rf "$ATLASES_DIR"
rm -rf "$CHECKPOINT_DIR"
rm -rf "$RGB_ATLASES_DIR"
rm -rf "$ALPHA_ATLASES_OUTPUT_DIR"
rm -rf "$FINAL_RGBA_DIR"
rm -rf "$COMPRESSED_RGB_DIR"
rm -rf "$PREP_RGB_DIR"
rm -rf "$PREP_MASK_DIR"
