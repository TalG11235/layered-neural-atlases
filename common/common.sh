#!/usr/bin/env bash
##############################################################################
# Project-wide helper functions and one-time Mamba/Conda initialization.
##############################################################################

# ---- Pick the solver: honor CONDA_BIN if set, else prefer mamba, else conda ----
if [ -n "${CONDA_BIN:-}" ]; then
    CMD="$CONDA_BIN"                            # user override via env var
elif command -v mamba >/dev/null 2>&1; then
    CMD=mamba                                   # use mamba if available
else
    CMD=conda                                   # fall back to conda
fi
# ------------------------------------------------------------------------------

# ---- Load the chosen shell hook exactly once --------------------------------
# If a 'conda' (or 'mamba') function is already defined, skip reloading.
if ! declare -F conda >/dev/null 2>&1; then
    if command -v "$CMD" >/dev/null 2>&1; then
        eval "$($CMD shell.bash hook)"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        # as a last resort, hard-code Conda’s install path
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "[common.sh] ERROR: Neither '$CMD' nor 'conda' was found on PATH." >&2
        return 1
    fi
fi
# ------------------------------------------------------------------------------

# ------- Helper functions --------------------------------------------------
# Log the start of a job with timestamp and script name
log_start() {
    printf '%s [%s] Job started on %s\n' \
           "$(date +%F_%T)" "$0" "$(hostname)"
}

# Return the directory that *called* this helper (not common.sh itself)
get_script_dir() {
    local src="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    cd "$( dirname "$src" )" &>/dev/null && pwd
}
