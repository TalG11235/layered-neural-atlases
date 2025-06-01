#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --gres=gpu:2

#SBATCH --output=logs/%x-%j.out
rm -rf compress_results/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate compression

echo "Job started on $(hostname)"
# Run your training
python3 DiffEIC/inference_partition.py \
--ckpt_sd ./weight/v2-1_512-ema-pruned.ckpt \
--ckpt_lc ./weight/lc.ckpt \
--config DiffEIC/configs/model/diffeic.yaml \
--input atlases_test/ \
--output compress_results/ \
--steps 50 \
--device cuda 
