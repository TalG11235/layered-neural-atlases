#!/usr/bin/env bash
set -euo pipefail

##############################################################################
#  create-missing Conda environments from all *.yml / *.yaml files found in
#  a directory.  File name (minus extension) = env name.
#
#  usage:  
#   ./scripts/setup_envs.sh  # looks in ../conda_envs by default
#   ./scripts/setup_envs.sh other_specs/  # override folder
##############################################################################

# absolute path to the directory where this script itself sits
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# spec folder: 1st CLI arg **or** "../conda_envs" relative to the script
SPEC_DIR="${1:-$SCRIPT_DIR/../conda_envs}"

# abort early if folder is missing
[[ -d "$SPEC_DIR" ]] || {
  echo "Error: spec folder '$SPEC_DIR' not found." >&2
  exit 1
}

# helper: true (exit 0) if env exists
env_exists () {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

# iterate over every .yml / .yaml spec in the folder
shopt -s nullglob              # pattern that matches nothing -> empty array
for spec in "$SPEC_DIR"/*.yml "$SPEC_DIR"/*.yaml; do
  [[ -e "$spec" ]] || continue # nothing matched, skip loop

  env_name="$(basename "${spec%.*}")"   # strip path & extension

  if env_exists "$env_name"; then
    echo "✓ Conda env '$env_name' already exists – skipping."
  else
    echo "Creating env '$env_name' from '$spec' ..."
    conda env create -n "$env_name" -f "$spec"
  fi
done
