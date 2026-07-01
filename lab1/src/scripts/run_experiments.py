#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# 六个版本与实验要求中的表格一一对应。
# 每个命令都输出统一的 key=value 文本，便于后续自动汇总。
VERSIONS = [
    {"name": "1 Python", "cmd": [sys.executable, "python/matmul.py", "{n}", "{repeat}"]},
    {"name": "2 C/C++", "cmd": ["bin/bench_o0", "ijk", "{n}", "{repeat}"]},
    {"name": "3 调整循环顺序", "cmd": ["bin/bench_loop", "ikj", "{n}", "{repeat}"]},
    {"name": "4 编译优化", "cmd": ["bin/bench_opt", "ijk", "{n}", "{repeat}"]},
    {"name": "5 循环展开", "cmd": ["bin/bench_unroll", "unroll4", "{n}", "{repeat}"]},
    {
        "name": "6 Intel MKL",
        "cmd": ["bin/mkl_bench", "{n}", "{repeat}"],
        # 强制 MKL 走单线程，避免它因为默认多线程而与前面 5 个串行版本失去可比性。
        "env": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_THREADING_LAYER": "SEQUENTIAL"},
    },
]


def parse_kv(output: str):
    data = {}
    for line in output.strip().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


# 执行单个版本并解析输出。
def run_command(cmd, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_kv(completed.stdout)


# 优先复用 Makefile；只有在 make 不存在时才走脚本内置的 gcc 回退路径。
def ensure_build():
    if shutil.which("make"):
        subprocess.run(["make", "all"], cwd=ROOT, check=True)
        return

    cc = os.environ.get("CC", "gcc")
    cflags_o0 = ["-std=c11", "-Wall", "-Wextra", "-I./src", "-O0"]
    cflags_o3 = ["-std=c11", "-Wall", "-Wextra", "-I./src", "-O3", "-march=native"]
    common = ["src/common.c"]
    kernels = ["src/kernels.c"]
    bench = ["src/bench.c"]
    mkl = ["src/mkl_bench.c"]

    (ROOT / "bin").mkdir(exist_ok=True)

    commands = [
        [cc, *cflags_o0, "-o", "bin/bench_o0", *common, *kernels, *bench, "-lm"],
        [cc, *cflags_o0, "-o", "bin/bench_loop", *common, *kernels, *bench, "-lm"],
        [cc, *cflags_o3, "-o", "bin/bench_opt", *common, *kernels, *bench, "-lm"],
        [cc, *cflags_o3, "-o", "bin/bench_unroll", *common, *kernels, *bench, "-lm"],
        [
            cc,
            *cflags_o3,
            "-o",
            "bin/mkl_bench",
            *common,
            *mkl,
            "-L/opt/intel/oneapi/mkl/2025.3/lib",
            "-Wl,-rpath,/opt/intel/oneapi/mkl/2025.3/lib",
            "-lmkl_intel_lp64",
            "-lmkl_sequential",
            "-lmkl_core",
            "-lpthread",
            "-lm",
            "-ldl",
        ],
    ]

    for cmd in commands:
        subprocess.run(cmd, cwd=ROOT, check=True)


# 用 checksum 做快速正确性检查。
# 这里默认以 2 C/C++ 为基准，只要所有版本输出一致，就说明核心计算逻辑没有跑偏。
def correctness_check(n: int, repeat: int, tolerance: float):
    checksums = {}
    for version in VERSIONS:
        data = run_command([part.format(n=n, repeat=repeat) for part in version["cmd"]], version.get("env"))
        checksums[version["name"]] = float(data["checksum"])

    baseline = checksums["2 C/C++"]
    for name, checksum in checksums.items():
        if abs(checksum - baseline) > tolerance:
            raise SystemExit(f"correctness check failed: {name} checksum {checksum} vs baseline {baseline}")


# 为单个矩阵规模汇总六个版本的最终指标。
def summarize_for_size(n: int, repeat: int, peak_gflops: float):
    rows = []
    previous_runtime = None
    python_runtime = None

    for version in VERSIONS:
        data = run_command([part.format(n=n, repeat=repeat) for part in version["cmd"]], version.get("env"))
        runtime = float(data["median_sec"])
        gflops = float(data["gflops"])
        if python_runtime is None:
            python_runtime = runtime
        # 相对加速比：相对于上一版本的改进幅度。
        relative_speedup = previous_runtime / runtime if previous_runtime else 1.0
        # 绝对加速比：始终与 Python 基线比较。
        absolute_speedup = python_runtime / runtime if python_runtime else 1.0
        peak_percent = (gflops / peak_gflops * 100.0) if peak_gflops > 0.0 else 0.0
        rows.append(
            {
                "size": n,
                "version": version["name"],
                "runtime_sec": runtime,
                "relative_speedup": relative_speedup,
                "absolute_speedup": absolute_speedup,
                "gflops": gflops,
                "peak_percent": peak_percent,
            }
        )
        previous_runtime = runtime

    return rows


def write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "results.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "size",
                "version",
                "runtime_sec",
                "relative_speedup",
                "absolute_speedup",
                "gflops",
                "peak_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


# 额外导出一份 Markdown，方便直接粘到报告或 README 中查看。
def write_markdown(rows, peak_gflops: float):
    path = RESULTS_DIR / "results.md"
    grouped = {}
    for row in rows:
        grouped.setdefault(row["size"], []).append(row)

    lines = []
    lines.append("# 矩阵乘法对比实验结果")
    lines.append("")
    lines.append(f"- 理论单线程峰值性能假设：{peak_gflops:.2f} GFLOPS")
    lines.append("- 指标定义：GFLOPS = 2*N^3 / runtime / 1e9")
    lines.append("")

    for size, size_rows in grouped.items():
        lines.append(f"## N = {size}")
        lines.append("")
        lines.append("| 版本 | 运行时间(sec.) | 相对加速比 | 绝对加速比 | 浮点性能(GFLOPS) | 峰值性能百分比 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in size_rows:
            lines.append(
                f"| {row['version']} | {row['runtime_sec']:.6f} | {row['relative_speedup']:.3f} | {row['absolute_speedup']:.3f} | {row['gflops']:.3f} | {row['peak_percent']:.2f}% |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 384])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--check-size", type=int, default=64)
    parser.add_argument("--check-repeat", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--peak-gflops", type=float, default=8.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    # 允许用户显式指定 Python 解释器，兼容 conda/base 等环境。
    VERSIONS[0]["cmd"][0] = args.python

    ensure_build()
    correctness_check(args.check_size, args.check_repeat, args.tolerance)
    if args.check_only:
        print("correctness check passed")
        return

    rows = []
    for n in args.sizes:
        rows.extend(summarize_for_size(n, args.repeat, args.peak_gflops))

    csv_path = write_csv(rows)
    md_path = write_markdown(rows, args.peak_gflops)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
