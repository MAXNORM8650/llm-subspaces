#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 CONFIG.json" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" || ! -e /dev/kfd && ! -e /dev/nvidia0 ]]; then
    echo "Refusing to run: enter a Slurm GPU allocation first." >&2
    exit 2
fi

python -m subspace_gguf pipeline \
    --config "$1" \
    --stages atlas,analyze,report \
    --resume
