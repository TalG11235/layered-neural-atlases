#!/usr/bin/env bash
##############################################################################
# Project-wide helper functions and one-time Conda initialization.
##############################################################################

# ------- Conda initialization (run once per shell) -------------------------
# If the `conda` *function* is not yet available in this shell, load it
# via the official hook.  Works no matter where Miniconda/Anaconda lives.
if ! declare -F conda >/dev/null 2>&1; then
    # make sure the executable is on PATH; fall back to a fixed location
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        # last-resort hard-coded path
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "[common.sh] ERROR: Conda executable not found." >&2
        return 1
    fi
fi
# ---------------------------------------------------------------------------

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
