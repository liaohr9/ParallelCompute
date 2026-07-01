#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results
make all

./code/bin/hello_world 2 3 4 > results/hello_2_3_4.txt
./code/bin/hello_world 4 5 6 > results/hello_4_5_6.txt
./code/bin/matrix_transpose --demo --n 8 --variant tiled_pad --config 32x8 \
  > results/demo_matrix_8.txt
./code/bin/matrix_transpose --test > results/correctness.txt

tail -n 1 results/correctness.txt
