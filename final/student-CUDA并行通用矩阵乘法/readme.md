# CUDA 并行通用矩阵乘法复现说明

## 1. 项目说明

本压缩包用于复现 CUDA 并行通用矩阵乘法实验。实验实现并对比了 4 个 CUDA kernel：

- `global2d`：二维任务划分，每个线程计算一个输出元素。
- `global1d`：一维线性任务划分，将输出矩阵展平成一维数组。
- `global2d_btrans`：先转置矩阵 `B`，再进行二维任务划分。
- `shared_tiled`：使用 shared memory 的分块矩阵乘法。

正式实验共 200 组配置，覆盖 8 种矩阵规模、7 种二维 block、4 种一维 block 和 4 类 kernel。结果文件已经保存在 `code/result` 中。

## 2. 目录结构

```text
.
├── StudentID+Student+CUDA并行通用矩阵乘法.pdf
├── readme.md
└── code
    ├── source
    │   ├── matrix_mul.cu
    │   └── Makefile
    ├── script
    │   ├── run_experiments.py
    │   └── plot_results.py
    └── result
        ├── matrix_mul_results_20260615_003634.csv
        ├── matrix_mul_results_20260615_003634.md
        ├── report_summary_20260615_003634.md
        └── figures
```

## 3. 环境要求

- Linux
- NVIDIA GPU
- CUDA Toolkit / `nvcc`
- Python 3
- Python 包：`matplotlib`

本实验原始运行环境为 NVIDIA GeForce RTX 5090，CUDA 13.0，编译架构为 `sm_120`。如果复现机器的 GPU 架构不同，需要修改 `code/source/Makefile` 中的 `CUDA_ARCH`，例如改为 `sm_86`、`sm_89` 等。

## 4. 编译方法

在压缩包解压后的根目录执行：

```bash
make -C code/source
```

编译成功后会生成：

```text
code/source/matrix_mul
```

## 5. 单组实验复现

示例：运行 `global2d`，矩阵规模为 `1024 x 1024 x 1024`，block 为 `32x8`：

```bash
code/source/matrix_mul \
  --kernel global2d \
  --m 1024 --n 1024 --k 1024 \
  --block-x 32 --block-y 8 --tile-k 32 \
  --warmup 4 --repeats 12 --verify-samples 64
```

程序会输出一行 CSV 格式结果，包括平均时间、GFLOP/s、正确性校验结果和最大误差。

## 6. 批量实验复现

完整复现 200 组正式实验：

```bash
python3 code/script/run_experiments.py \
  --binary code/source/matrix_mul \
  --output-dir code/result \
  --warmup 4 \
  --repeats 12 \
  --verify-samples 64
```

该命令会在 `code/result` 中生成新的 CSV 和 Markdown 结果文件。由于 GPU 型号、驱动版本、CUDA 版本和当前系统负载不同，复现实验的运行时间和 GFLOP/s 可能与报告中的数值略有差异。

快速功能验证可执行：

```bash
python3 code/script/run_experiments.py \
  --binary code/source/matrix_mul \
  --output-dir code/result \
  --quick \
  --warmup 1 \
  --repeats 1 \
  --verify-samples 8
```

## 7. 图表复现

根据正式 CSV 重新生成 matplotlib 图：

```bash
python3 code/script/plot_results.py \
  --csv code/result/matrix_mul_results_20260615_003634.csv \
  --output-dir code/result/figures
```

生成的图片包括不同 kernel 性能对比、不同矩阵规模性能、不同 block 大小性能、正确性误差等。

## 8. 已包含结果

`code/result/matrix_mul_results_20260615_003634.csv` 是报告使用的正式实验原始数据，共 200 组实验。所有实验均通过 CPU 抽样正确性校验。

最终报告 PDF 位于压缩包根目录：

```text
StudentID+Student+CUDA并行通用矩阵乘法.pdf
```
