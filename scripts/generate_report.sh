#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 ANALYSIS.json OUTPUT_DIR" >&2
    exit 2
fi

analysis=$1
output_dir=$2
mkdir -p "$output_dir"

python -m subspace_gguf report \
    --analysis "$analysis" \
    --output-json "$output_dir/subspace_report.json" \
    --output-markdown "$output_dir/SUBSPACE_EFFECTIVENESS.md"
