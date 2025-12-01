#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --gres=gpu:2

#SBATCH --output=logs/%x-%j.out

SCRIPT_DIR="$(pwd)"
ATLASES_DIR=$SCRIPT_DIR/atlases_test
OUTPUT_DIR=$SCRIPT_DIR/compressed_atlases
RDEIC_DIR=$SCRIPT_DIR/thirdparty/RDEIC
WEIGHT_DIR=$RDEIC_DIR/weight

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdeic

cd $RDEIC_DIR

echo $WEIGHT_DIR

echo "Job started on $(hostname)"
# Run your training
python3 inference_partition.py \
--ckpt_sd $WEIGHT_DIR/v2-1_512-ema-pruned.ckpt \
--ckpt_cc $WEIGHT_DIR/rdeic_2_step2.ckpt \
--config configs/model/rdeic.yaml \
--input $ATLASES_DIR \
--output $OUTPUT_DIR \
--steps 2 \
--guidance_scale 1 \
--device cuda 
