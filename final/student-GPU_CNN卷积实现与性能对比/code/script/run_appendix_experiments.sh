#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$CODE_DIR/source"
APP_DIR="$SOURCE_DIR/appendix"
MAIN_RESULT_DIR="$CODE_DIR/result/main"
APP_RESULT_DIR="$CODE_DIR/result/appendix"
APP_FIGURE_DIR="$APP_RESULT_DIR/figures"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDNN_HOME="${CUDNN_HOME:-/home/haoran/miniconda3/envs/act/lib/python3.10/site-packages/nvidia/cudnn}"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$SOURCE_DIR/results" "$APP_RESULT_DIR" "$APP_FIGURE_DIR"

# 附录分析脚本会读取主实验 CSV，用它来判断附录方法是否超过主实验最优值。
if [[ -f "$MAIN_RESULT_DIR/experiment_results.csv" ]]; then
    cp "$MAIN_RESULT_DIR/experiment_results.csv" "$SOURCE_DIR/results/experiment_results.csv"
fi

cd "$APP_DIR"
bash scripts/run_appendix_experiments.sh

result_files=(results/*.csv results/*.tex)
if ((${#result_files[@]})); then
    cp "${result_files[@]}" "$APP_RESULT_DIR"/
fi

figure_files=(img/*.png)
if ((${#figure_files[@]})); then
    cp "${figure_files[@]}" "$APP_FIGURE_DIR"/
fi

echo "Appendix experiment results: $APP_RESULT_DIR"
