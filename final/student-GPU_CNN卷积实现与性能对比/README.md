# GPU CNN 卷积实现与性能对比复现说明

本压缩包用于提交和复现实验。根目录包含最终报告 PDF、可复现实验代码、运行脚本和已经得到的结果文件。

## 目录结构

```text
.
├── StudentID+Student+GPU_CNN卷积实现与性能对比.pdf
├── README.md
└── code
    ├── source
    │   ├── Makefile
    │   ├── src
    │   │   ├── conv_benchmark.cu
    │   │   └── cuda_utils.h
    │   ├── scripts
    │   │   ├── run_experiments.sh
    │   │   └── summarize_results.py
    │   └── appendix
    │       ├── Makefile
    │       ├── src
    │       │   └── optimization_benchmark.cu
    │       └── scripts
    │           ├── run_appendix_experiments.sh
    │           └── analyze_appendix.py
    ├── script
    │   ├── run_main_experiments.sh
    │   ├── run_appendix_experiments.sh
    │   ├── summarize_results.py
    │   └── analyze_appendix_original.py
    └── result
        ├── main
        └── appendix
```

`code/source` 保存源码和原始实验脚本；`code/script` 保存适配本提交包目录结构的运行入口；`code/result` 保存已生成的 CSV、表格和图表结果。压缩包中未包含 `bin/` 可执行文件、LaTeX 中间文件、checkpoint 等可重新生成或体积较大的文件。

## 环境要求

推荐环境与本次实验一致：

- Linux x86_64
- NVIDIA GPU，实验机器为 RTX 5090
- CUDA Toolkit，默认路径 `/usr/local/cuda`
- cuBLAS
- cuDNN v9，默认路径为 `/home/haoran/miniconda3/envs/act/lib/python3.10/site-packages/nvidia/cudnn`
- `nvcc`、`make`、`bash`、`python3`
- Python 可选依赖：`matplotlib`，用于重新生成附录图表

默认编译架构为 `sm_120`。如果复现机器 GPU 架构不同，可通过环境变量覆盖，例如：

```bash
CUDA_ARCH=sm_89 bash code/script/run_main_experiments.sh
```

如果 CUDA 或 cuDNN 不在默认路径，可显式指定：

```bash
CUDA_HOME=/usr/local/cuda \
CUDNN_HOME=/path/to/cudnn \
bash code/script/run_main_experiments.sh
```

## 复现主实验

在压缩包根目录执行：

```bash
bash code/script/run_main_experiments.sh
```

该脚本会完成以下工作：

1. 进入 `code/source`；
2. 编译 `src/conv_benchmark.cu`；
3. 遍历输入尺寸、stride 和输出通道数；
4. 运行 CUDA 直接卷积、im2col + cuBLAS、cuDNN 三种方法；
5. 将结果复制到 `code/result/main`。

完整实验规模较大。如需先做快速检查，可以缩小实验矩阵：

```bash
SIZES="256" STRIDES="1" OUT_CHANNELS="1" \
bash code/script/run_main_experiments.sh
```

## 复现附录优化实验

在压缩包根目录执行：

```bash
bash code/script/run_appendix_experiments.sh
```

该脚本会编译并运行 `code/source/appendix/src/optimization_benchmark.cu`，测试以下方法：

- `direct_baseline`
- `shared_tile`
- `register_block2`
- `cross_channel4`
- `implicit_im2col`
- `tensor_core_tf32`

附录脚本会把结果复制到 `code/result/appendix`，并在安装 `matplotlib` 时重新生成附录图表到 `code/result/appendix/figures`。

## 结果文件说明

主实验结果位于 `code/result/main`：

- `experiment_results.csv`：主实验最新完整结果；
- `experiment_results_*.csv`：运行过程中的时间戳结果备份；
- `experiment_results.md`：由 CSV 整理出的 Markdown 结果；
- `figures/*.png`：报告中使用的主实验图表。

附录结果位于 `code/result/appendix`：

- `appendix_optimization.csv`：附录优化实验最新完整结果；
- `summary_by_method.csv`：各优化方法总体统计；
- `auto_selector_rule.csv`：按输出通道和 stride 得到的选择规则；
- `auto_selector_summary.csv`：自动选择器统计；
- `improvements_vs_main_best.csv`：相对主实验最优结果仍有提升的 case；
- `generated_tables.tex`：报告附录中使用的自动生成表格；
- `figures/*.png`：附录图表。

## 清理与重新编译

主实验清理：

```bash
make -C code/source clean
```

附录实验清理：

```bash
make -C code/source/appendix clean
```

清理只会删除重新编译得到的可执行文件，不会删除 `code/result` 中已经保存的实验结果。
