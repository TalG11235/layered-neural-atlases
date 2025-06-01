#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --gres=gpu:2

#SBATCH --output=logs/%x-%j.out

echo "Job started on $(hostname)"
# Run your training
source ~/miniconda3/etc/profile.d/conda.sh

conda activate neural_atlases

CONFIG_PATH=config.json
INPUT_CKPT=checkpoint/checkpoint
OUTPUT_CKPT=test/checkpoint_quantized

python quantize_checkpoint.py $CONFIG_PATH $INPUT_CKPT $OUTPUT_CKPT