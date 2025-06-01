#!/bin/bash
#SBATCH --job-name=neural-atlas-train
#SBATCH --gres=gpu:2

#SBATCH --output=logs/%x-%j.out

echo "Job started on $(hostname)"
# Run your training
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
