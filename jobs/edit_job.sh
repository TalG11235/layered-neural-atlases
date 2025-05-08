#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --gres=gpu:2

#SBATCH --output=logs/%x-%j.out

echo "Job started on $(hostname)"
# Run your training
rm -rf compressed_editing_outputs/

source ~/miniconda3/etc/profile.d/conda.sh

conda activate neural_atlases

python only_edit.py --trained_model_folder=test/ --video_name=blackswan --output_folder=quantize_editing_outputs --edit_foreground_path=atlases/texture_orig1.png --edit_background_path=atlases/texture_orig2.png

