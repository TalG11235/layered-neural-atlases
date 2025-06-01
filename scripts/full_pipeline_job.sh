#!/bin/env bash

set -euo pipefail

# ------- Step 0: Setup ----------------------------------------------------------------------
# Declare the path to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Source the common functions
source "$SCRIPT_DIR/../common/common.sh"

# Remove any previous results, if those exist
# rm -rf results/ atlases/ compressed_atlases/ compressed_results/ compressed_editing_outputs/

# Create output directories
# mkdir -p results atlases compressed_atlases compressed_results compressed_editing_outputs

# Create missing conda environments
setup_envs

# log the start of the job
log_start
# -------------------------------------------------------------------------------------------

# ------- Step 1: Get a video to run on -----------------------------------------------------
# This could mean one of two things:
# 1. Download the video as a folder, separated by frames (From the DAVIS dataset)
# 2. Download the video as a single file, and separate it into frames
# Either way, at the end of it, we should have a folder named "data" with frames in it.

# -------------------------------------------------------------------------------------------

# ------- Step 2: Preprocessing -------------------------------------------------------------
# Run the two preprocessing scripts:
# 1. `preprocess_mask_rcnn.py` to segement the video to front and back
# 2. `preprocess_optical_flow.py` to get the optical flow of the video using RAFT

# -------------------------------------------------------------------------------------------

# ------- Step 3: Run the training job ------------------------------------------------------
# This is as simple as activating the neural_atlases conda environment and running the 
# training script, which is train.py.
# -------------------------------------------------------------------------------------------

# ------- Step 4: Extract relevant data from training result --------------------------------
# After the training is done, we need to extract the texture files from the results.
# This is done by finding the latest subdirectory in the results directory, and copying the
# texture files to the atlases directory.
# -------------------------------------------------------------------------------------------

# ------- Step 5: Compress the atlases ------------------------------------------------------
# Compress the atlases using RDEIC, and move the compressed atlases to the
# compressed_atlases directory.
# -------------------------------------------------------------------------------------------

# ------- Step 6: Save Checkpoint -----------------------------------------------------------
# Save the checkpoint of the training job to a file named checkpoint.pth in the checkpoint 
# directory.
# -------------------------------------------------------------------------------------------

# ------- Step 7: Add alpha channels to atlases ---------------------------------------------
# \
# -------------------------------------------------------------------------------------------

# ------- Step 8: Quantize ------------------------------------------------------------------
# Quantize the atlases using the quantization script, and move the quantized atlases to the
# compressed_editing_outputs directory.
# -------------------------------------------------------------------------------------------

# ------- Step 9: Reconstruct the video -----------------------------------------------------
# Reconstruct the video using the editing script, and save the output to the
# compressed_editing_outputs directory.
# -------------------------------------------------------------------------------------------