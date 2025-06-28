#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --gres=gpu:2

#SBATCH --output=logs/%x-%j.out

echo "Job started on $(hostname)"
python src/preprocess_mask_rcnn.py --vid-path data/ShakeNDry/ShakeNDry --class_name dog