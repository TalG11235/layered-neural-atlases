#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --nodelist=lambda3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

#SBATCH --output=logs/%x-%j.out

echo "Job started on $(hostname)"

#### This script is meant to be used from the Layered neural atlases project directory ####

# general variables for the pipline
PROJECT_DIR="$(pwd)"
PROJECT_SRC_DIR="$PROJECT_DIR/src"
EXPERIMENTS_DIR="$PROJECT_DIR/experiments"
EXPERIMENTS_RESULTS_DIR="$EXPERIMENTS_DIR/results"
ATLASES_DIR="$EXPERIMENTS_DIR/atlases"
OUTPUT_DIR="$EXPERIMENTS_DIR/compressed_atlases"
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

# clear previous intermidiate directories
rm -rf "$ATLASES_DIR"
rm -rf "$OUTPUT_DIR"
rm -rf "$CHECKPOINT_DIR"
rm -rf "$RGB_ATLASES_DIR"
rm -rf "$ALPHA_ATLASES_OUTPUT_DIR"

##################################### Training ################################################
echo "Training"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural_atlases

python $PROJECT_SRC_DIR/train.py $CONFIG_PATH

# Navigate into the latest directory inside results/
exp_dir=$(find results/ -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)
mv $exp_dir $EXPERIMENTS_RESULTS_DIR
exp_dir_name=$(basename "$exp_dir")
rm -rf results

################################ Atlases Compression ##########################################
# Find the subdirectory with the highest number (frame index)
latest_experiment="$EXPERIMENTS_RESULTS_DIR/$(basename "$exp_dir")"
latest_experiment_evaluation=$(find "$latest_experiment" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)

# Make sure the output folder exists
mkdir -p "$ATLASES_DIR"
# Copy the two texture files to atlases/
cp "$latest_experiment_evaluation/texture_orig1.png" "$latest_experiment_evaluation/texture_orig2.png" $ATLASES_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdeic

cd $RDEIC_DIR

python3 $RDEIC_DIR/inference_partition.py \
--ckpt_sd $WEIGHT_DIR/v2-1_512-ema-pruned.ckpt \
--ckpt_cc $WEIGHT_DIR/rdeic_2_step2.ckpt \
--config configs/model/rdeic.yaml \
--input $ATLASES_DIR \
--output $OUTPUT_DIR \
--steps 2 \
--guidance_scale 1 \
--device cuda 

cd $PROJECT_DIR

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

##################################### Atlases Alpha Layers ################################################
echo "Atlases Alpha Layers"
# add alpha channels to atlases
source ~/miniconda3/etc/profile.d/conda.sh

conda activate neural_atlases

mkdir -p "$ALPHA_ATLASES_OUTPUT_DIR"

for file in "$RGB_ATLASES_DIR"/*.png; do
    filename=$(basename "$file")
    output_file="$ALPHA_ATLASES_OUTPUT_DIR/$filename"

    echo "Processing $filename"

    python3 - <<EOF
import imageio
import numpy as np

rgb = imageio.imread("$file").astype(np.float32) / 255.0

# Check if image already has alpha
if rgb.ndim == 3 and rgb.shape[2] == 4:
    rgba = rgb
else:
    alpha = np.ones(rgb.shape[:2], dtype=np.float32)
    rgba = np.concatenate([rgb[:, :, :3], alpha[:, :, None]], axis=-1)

imageio.imwrite("$output_file", (rgba * 255).astype(np.uint8))
EOF

done
mkdir "$EXPERIMENTS_DIR/outputs/"
final_output_folder="$EXPERIMENTS_DIR/outputs/$exp_dir_name"
# copy everything to one folder
mkdir -p "$final_output_folder"

cp $CONFIG_PATH $ALPHA_ATLASES_OUTPUT_DIR/texture_orig1.png $ALPHA_ATLASES_OUTPUT_DIR/texture_orig2.png $final_output_folder

##################################### Checkpoint Quiantization ################################################
echo "Checkpoint Quiantization"
# quantization
source ~/miniconda3/etc/profile.d/conda.sh

conda activate neural_atlases

python $PROJECT_SRC_DIR/quantize_checkpoint.py $CONFIG_PATH $CHECKPOINT_DIR/checkpoint $final_output_folder/checkpoint

##################################### Video Reconstruction ################################################
echo "Video Reconstruction"
# recounstruct video with compressed atlases
python $PROJECT_SRC_DIR/only_edit.py --trained_model_folder=$final_output_folder/ \
--video_name=blackswan \
--output_folder="$final_output_folder/reconstruction/" \
--edit_foreground_path="$final_output_folder"/texture_orig1.png \
--edit_background_path="$final_output_folder"/texture_orig2.png




