#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --gres=gpu:2

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

mkdir -p compressed_results

cp $exp_dir/config.json compressed_atlases/texture_orig1.png compressed_atlases/texture_orig2.png compressed_results/


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




