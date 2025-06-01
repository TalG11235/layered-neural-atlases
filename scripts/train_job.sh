#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --gres=gpu:2

#SBATCH --output=logs/%x-%j.out

echo "Job started on $(hostname)"
# Run your training
source ~/miniconda3/etc/profile.d/conda.sh

conda activate neural_atlases

python ./src/layered-neural-atlases/train.py ./config/train_configs/config.json
