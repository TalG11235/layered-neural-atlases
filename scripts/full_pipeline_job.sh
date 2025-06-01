#!/bin/env bash

set -euo pipefail

# ------- Setup -----------------------------------------------------------------------------
# Declare the path to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Source the common functions
source "$SCRIPT_DIR/../common/common.sh"

# Remove any previous results, if those exist
rm -rf results/ atlases/ compressed_atlases/ compressed_results/ compressed_editing_outputs/

# Create output directories
mkdir -p results atlases compressed_atlases compressed_results compressed_editing_outputs

# Create missing conda environments
setup_envs

# log the start of the job
log_start
# -------------------------------------------------------------------------------------------