#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
source "$SCRIPT_DIR/../common/common.sh"   # sets CMD and loads $CMD shell hook

ENV_NAME=${1:-neural_atlases}
YAML_FILE=${2:-$SCRIPT_DIR/../envs/${ENV_NAME}.yml}

echo "Updating env '$ENV_NAME' with $CMD from '$YAML_FILE'"
$CMD env update -n "$ENV_NAME" -f "$YAML_FILE" --prune
