#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$CODE_DIR/source"
RESULT_DIR="$CODE_DIR/result/main"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDNN_HOME="${CUDNN_HOME:-/home/haoran/miniconda3/envs/act/lib/python3.10/site-packages/nvidia/cudnn}"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$RESULT_DIR" "$SOURCE_DIR/results"

cd "$SOURCE_DIR"
bash scripts/run_experiments.sh

result_files=(results/experiment_results*.csv)
if ((${#result_files[@]})); then
    cp "${result_files[@]}" "$RESULT_DIR"/
fi

python3 scripts/summarize_results.py results/experiment_results.csv "$RESULT_DIR/experiment_results.md"

echo "Main experiment results: $RESULT_DIR"
