# ParallelCompute

并行程序设计实验与期末项目整理。

## Repository Structure

| Directory | Topic | Main Contents |
| --- | --- | --- |
| `lab1/` | 环境配置与串行矩阵乘法 | C 源码、脚本、实验结果、报告 |
| `lab2/` | MPI 并行矩阵乘法 | MPI 矩阵乘法、Cannon 算法、稀疏矩阵实验、报告 |
| `lab3/` | MPI 矩阵乘法进阶 | 点对点通信、集合通信、列块划分、结构体通信、性能图 |
| `lab4/` | Pthreads 方程求解与蒙特卡洛 | Pthreads C 程序、可执行文件、实验报告 |
| `lab5/` | OpenMP 与 `parallel_for` | OpenMP GEMM、Pthreads GEMM、`parallel_for` 实现、结果表 |
| `lab6/` | Heated Plate 并行化 | Pthreads/OpenMP heated plate 实现、实验脚本、结果表 |
| `lab7/` | MPI/OpenMP 综合实验 | FFT、MPI/OpenMP 性能实验、结果图和统计表 |
| `lab8/` | OpenMP 最短路径 | `shortest_path_omp.cpp`、benchmark 脚本、查询与输出结果 |
| `lab9/` | CUDA 基础实验 | CUDA hello world、矩阵转置、GPU 信息、正确性和性能结果 |
| `final/` | 期末项目 | CUDA 通用矩阵乘法、GPU CNN 卷积实现与性能对比 |

## Final Projects

`final/` 下包含两个独立项目：

- `student-CUDA并行通用矩阵乘法/`
  - CUDA 矩阵乘法源码
  - 实验脚本
  - CSV/Markdown 结果
  - 性能图
  - 项目报告 PDF
- `student-GPU_CNN卷积实现与性能对比/`
  - CUDA CNN 卷积 benchmark
  - main/appendix 两组实验结果
  - 性能图与 LaTeX 表格
  - 项目报告 PDF

## Running

每个实验目录保留原始提交中的源码、脚本、结果和报告。运行方式以各目录内的 `README.md`、`readme.md`、`Makefile`、`run_*.sh` 或实验脚本为准。

常见依赖包括：

- C/C++ 编译器
- MPI
- OpenMP
- CUDA Toolkit
- Python 3

不同实验依赖不完全相同，建议进入对应 `lab*` 或 `final/*` 目录后再查看脚本和 README。

## Cleanup Policy

原始压缩包未提交到仓库；仓库只保留解压后的可查看文件。含个人标识符的报告 PDF 已替换为匿名占位 PDF。

已排除：

- `.DS_Store`
- `__MACOSX/`
- `._*`
- `.claude/`
- `__pycache__/`
- `*.zip`
