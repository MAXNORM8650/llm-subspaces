#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 CONFIG.json [STAGES]" >&2
    echo "stages default: atlas,analyze,report,imatrix,quantize,smoke" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" || ! -e /dev/kfd && ! -e /dev/nvidia0 ]]; then
    echo "Refusing to run: enter a Slurm GPU allocation first." >&2
    exit 2
fi

config=$1
stages=${2:-atlas,analyze,report,imatrix,quantize,smoke}

python -m subspace_gguf pipeline \
    --config "$config" \
    --stages "$stages" \
    --resume
