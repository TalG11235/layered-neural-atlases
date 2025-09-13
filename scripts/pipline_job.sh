#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --nodelist=lambda3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

#SBATCH --output=logs/%x-%j.out

# clear previous data
rm -rf results/
rm -rf atlases/
rm -rf compressed_atlases/
rm -rf compressed_results/
rm -rf compressed_editing_outputs/


echo "Job started on $(hostname)"
# Run training
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural_atlases

python train.py config.json

# Navigate into the only directory inside results/
exp_dir=$(find results/ -mindepth 1 -maxdepth 1 -type d | head -n 1)

# Find the subdirectory with the highest number (frame index)
latest_subdir=$(find "$exp_dir" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)

# Make sure the output folder exists
mkdir -p atlases

# # Copy the two texture files to atlases/
cp "$latest_subdir/texture_orig1.png" "$latest_subdir/texture_orig2.png" atlases/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate compression

# compress atlases and move to folder
python3 DiffEIC/inference_partition.py \
--ckpt_sd ./weight/v2-1_512-ema-pruned.ckpt \
--ckpt_lc ./weight/lc.ckpt \
--config DiffEIC/configs/model/diffeic.yaml \
--input atlases/ \
--output compressed_atlases/ \
--steps 50 \
--device cuda 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural_atlases

mkdir -p checkpoint

python <<EOF
import torch

checkpoint_path = "$exp_dir/checkpoint"
checkpoint = torch.load(checkpoint_path)

# Remove the specified keys
checkpoint.pop("F_atlas_state_dict", None)
checkpoint.pop("optimizer_all_state_dict", None)

# Save the new checkpoint to compressed_results
torch.save(checkpoint, "checkpoint/checkpoint")
EOF

# add alpha channels to atlases
source ~/miniconda3/etc/profile.d/conda.sh

conda activate neural_atlases

RGB_FOLDER=compressed_atlases/
OUTPUT_FOLDER=compressed_atlases_alpha/
ALPHA_FOLDER=$1

mkdir -p "$OUTPUT_FOLDER"

for file in "$RGB_FOLDER"/*.png; do
    filename=$(basename "$file")
    output_file="$OUTPUT_FOLDER/$filename"

    if [ -n "$ALPHA_FOLDER" ] && [ -f "$ALPHA_FOLDER/$filename" ]; then
        echo "Adding alpha from $ALPHA_FOLDER/$filename to $filename"
        python3 - <<EOF
import imageio
import numpy as np

rgb = imageio.imread("$file").astype(np.float32) / 255.0
alpha = imageio.imread("$ALPHA_FOLDER/$filename").astype(np.float32) / 255.0

# Use single-channel alpha if provided RGBA image
if alpha.ndim == 3:
    alpha = alpha[:, :, 3] if alpha.shape[2] == 4 else alpha[:, :, 0]

rgba = np.concatenate([rgb[:, :, :3], alpha[:, :, None]], axis=-1)
imageio.imwrite("$output_file", (rgba * 255).astype(np.uint8))
EOF

    else
        echo "No alpha image found. Adding constant alpha=1 to $filename"
        python3 - <<EOF
import imageio
import numpy as np

rgb = imageio.imread("$file").astype(np.float32) / 255.0
alpha = np.ones(rgb.shape[:2], dtype=np.float32)
rgba = np.concatenate([rgb[:, :, :3], alpha[:, :, None]], axis=-1)
imageio.imwrite("$output_file", (rgba * 255).astype(np.uint8))
EOF
    fi
done

# copy everything to one folder
mkdir -p compressed_results

cp $exp_dir/config.json compressed_atlases_alpha/texture_orig1.png compressed_atlases_alpha/texture_orig2.png compressed_results/

# quantization
source ~/miniconda3/etc/profile.d/conda.sh

conda activate neural_atlases

python quantize_checkpoint.py compressed_results/config.json checkpoint/checkpoint compressed_results/checkpoint

# recounstruct video with compressed atlases
python only_edit.py --trained_model_folder=compressed_results/ \
--video_name=blackswan \
--output_folder=compressed_editing_outputs \
--edit_foreground_path=compressed_results/texture_orig1.png \
--edit_background_path=compressed_results/texture_orig2.png

