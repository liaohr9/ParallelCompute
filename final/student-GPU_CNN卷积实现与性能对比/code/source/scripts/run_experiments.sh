#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CUDNN_HOME="${CUDNN_HOME:-/home/haoran/miniconda3/envs/act/lib/python3.10/site-packages/nvidia/cudnn}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDNN_HOME CUDA_HOME
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p results

SIZES="${SIZES:-256 512 1024 2048 4096}"
STRIDES="${STRIDES:-1 2 3}"
OUT_CHANNELS="${OUT_CHANNELS:-1 16 64 128}"

choose_iters() {
    local size="$1"
    local out_ch="$2"
    if (( size <= 256 )); then
        echo 80
    elif (( size <= 512 )); then
        echo 60
    elif (( size <= 1024 )); then
        echo 35
    elif (( size <= 2048 )); then
        if (( out_ch >= 64 )); then
            echo 18
        else
            echo 24
        fi
    else
        if (( out_ch >= 64 )); then
            echo 8
        else
            echo 12
        fi
    fi
}

choose_warmup() {
    local size="$1"
    if (( size <= 1024 )); then
        echo 8
    elif (( size <= 2048 )); then
        echo 5
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

make -j"$(nproc)"

timestamp="$(date +%Y%m%d_%H%M%S)"
raw_csv="results/experiment_results_${timestamp}.csv"
latest_csv="results/experiment_results.csv"

./bin/conv_benchmark --csv-header > "$raw_csv"

total_cases=0
for _oc in $OUT_CHANNELS; do
    for _size in $SIZES; do
        for _stride in $STRIDES; do
            if case_supported "$_size" "$_stride" "$_oc"; then
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
                echo "[skip] size=$size stride=$stride out_channels=$out_ch exceeds cuDNN tensor element limit"
                continue
            fi
            case_id=$((case_id + 1))
            iters="$(choose_iters "$size" "$out_ch")"
            warmup="$(choose_warmup "$size")"
            echo "[$case_id/$total_cases] size=$size stride=$stride out_channels=$out_ch warmup=$warmup iters=$iters"
            ./bin/conv_benchmark \
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

echo "Done."
echo "CSV: $latest_csv"
