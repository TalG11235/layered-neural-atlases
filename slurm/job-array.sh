#!/bin/bash
#SBATCH --job-name=svc-array
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-1

module load miniconda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate svc

python -m src.pipeline --cfg pipeline.yaml --slurm_idx $SLURM_ARRAY_TASK_ID
