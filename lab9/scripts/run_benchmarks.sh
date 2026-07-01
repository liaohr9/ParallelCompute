#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results 报告/img
make all

nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,power.limit \
  --format=csv,noheader > results/gpu_info.txt
nvcc --version > results/nvcc_version.txt

./code/bin/matrix_transpose --benchmark \
  --sizes 512,768,1024,1536,2048 \
  --configs 8x4,8x8,16x4,16x8,16x16,32x4,32x8,32x16,32x32,64x4,64x8,64x16 \
  --variants naive_write_strided,naive_read_strided,tiled_nopad,tiled_pad \
  --warmup 8 --repeat -1 --min-ms 350 \
  --csv results/transpose_assignment.csv

./code/bin/matrix_transpose --benchmark \
  --sizes 4096,8192,12288,16384,24576,32768 \
  --configs 32x8,32x16,64x8,64x16 \
  --variants naive_write_strided,naive_read_strided,tiled_pad \
  --warmup 5 --repeat -1 --min-ms 400 \
  --csv results/transpose_heavy.csv

python3 scripts/plot_results.py
