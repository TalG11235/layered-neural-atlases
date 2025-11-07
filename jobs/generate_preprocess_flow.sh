#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --gres=gpu:2

#SBATCH --output=logs/%x-%j.out

echo "Job started on $(hostname)"

source ~/miniconda3/etc/profile.d/conda.sh

conda activate neural_atlases

python src/preprocess_optical_flow.py --vid-path data/Frames/ReadySetGo --max_long_edge 768