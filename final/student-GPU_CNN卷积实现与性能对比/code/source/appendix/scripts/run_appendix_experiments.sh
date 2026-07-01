#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDNN_HOME="${CUDNN_HOME:-/home/haoran/miniconda3/envs/act/lib/python3.10/site-packages/nvidia/cudnn}"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p results img
make -j"$(nproc)"

SIZES="${SIZES:-256 512 1024 2048 4096}"
STRIDES="${STRIDES:-1 2 3}"
OUT_CHANNELS="${OUT_CHANNELS:-1 16 64 128}"

choose_iters() {
    local size="$1"
    local out_ch="$2"
    if (( size <= 256 )); then
        echo 60
    elif (( size <= 512 )); then
        echo 45
    elif (( size <= 1024 )); then
        echo 30
    elif (( size <= 2048 )); then
        if (( out_ch >= 64 )); then
            echo 14
        else
            echo 20
        fi
    else
        if (( out_ch >= 64 )); then
            echo 6
        else
            echo 10
        fi
    fi
}

choose_warmup() {
    local size="$1"
    if (( size <= 1024 )); then
        echo 6
    elif (( size <= 2048 )); then
        echo 4
    else
        echo 3
    fi
}

case_supported() {
    local size="$1"
    local stride="$2"
    local out_ch="$3"
    local out=$(( (size + stride - 1) / stride ))
    local elems=$(( out_ch * out * out ))
    (( elems < 2147483648 ))
}

timestamp="$(date +%Y%m%d_%H%M%S)"
raw_csv="results/appendix_optimization_${timestamp}.csv"
latest_csv="results/appendix_optimization.csv"

./bin/optimization_benchmark --csv-header > "$raw_csv"

total_cases=0
for out_ch in $OUT_CHANNELS; do
    for size in $SIZES; do
        for stride in $STRIDES; do
            if case_supported "$size" "$stride" "$out_ch"; then
                total_cases=$((total_cases + 1))
            fi
        done
    done
done

case_id=0
for out_ch in $OUT_CHANNELS; do
    for size in $SIZES; do
        for stride in $STRIDES; do
            if ! case_supported "$size" "$stride" "$out_ch"; then
                echo "[skip] size=$size stride=$stride out_channels=$out_ch exceeds 2^31 output elements"
                continue
            fi
            case_id=$((case_id + 1))
            iters="$(choose_iters "$size" "$out_ch")"
            warmup="$(choose_warmup "$size")"
            echo "[$case_id/$total_cases] size=$size stride=$stride out_channels=$out_ch warmup=$warmup iters=$iters"
            ./bin/optimization_benchmark \
                --size "$size" \
                --stride "$stride" \
                --out-channels "$out_ch" \
                --warmup "$warmup" \
                --iters "$iters" >> "$raw_csv"
            cp "$raw_csv" "$latest_csv"
        done
    done
done

cp "$raw_csv" "$latest_csv"
python3 scripts/analyze_appendix.py "$latest_csv"

echo "Done."
echo "CSV: $latest_csv"
echo "Tables: results/generated_tables.tex"
echo "Figures: img/"
