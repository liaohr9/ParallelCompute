#!/usr/bin/env python3
"""
稀疏矩阵乘法性能测试脚本

遍历不同稀疏度、进程数和矩阵规模，对比稀疏 vs 稠密方案的性能差异。

使用方法：
    ./benchmark_sparse.py
"""

import subprocess
import sys
import os

# ---- 测试参数 ----
PROCESSES = [1, 2, 4, 8, 16, 32]        # 进程数
SIZES = [128, 256, 512, 1024, 2048, 4096]       # 矩阵规模
SPARSITIES = [0.001, 0.01, 0.05, 0.1, 0.5]  # 稀疏度
RUNS = 5  # 每组重复运行次数

PROGRAM = "./mpi_sparse"

def check_program():
    if not os.path.exists(PROGRAM):
        print("程序未编译，正在执行 make...")
        result = subprocess.run(["make", "mpi_sparse"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"编译失败：\n{result.stderr}")
            sys.exit(1)

def run_test(np, size, sparsity):
    """运行一次测试，返回 (time_ms, nnz, theoretical_speedup)"""
    try:
        result = subprocess.run(
            ["mpirun", "-np", str(np), "--oversubscribe", PROGRAM, str(size), str(sparsity)],
            capture_output=True, text=True, timeout=3600
        )
        time_ms = None
        nnz = None
        theoretical_speedup = None
        for line in result.stdout.split('\n'):
            if '计算时间:' in line:
                # "计算时间: 0.002653 秒 (2.65 毫秒)"
                parts = line.split()
                for i, p in enumerate(parts):
                    if '计算时间:' in p and i + 3 < len(parts):
                        time_ms = float(parts[i + 3].lstrip('(').rstrip(')'))
                        break
            if 'A 的非零元素数 (nnz)' in line:
                parts = line.split(':')
                nnz = int(parts[-1].strip())
            if '理论加速比:' in line:
                parts = line.split(':')
                val = parts[-1].strip().replace('x', '')
                theoretical_speedup = float(val)
        return time_ms, nnz, theoretical_speedup
    except Exception:
        return None, None, None

def main():
    check_program()

    try:
        import multiprocessing
        max_procs = multiprocessing.cpu_count()
    except:
        max_procs = 8

    print(f"本机 CPU 逻辑核心数: {max_procs}")
    print(f"每组参数运行 {RUNS} 次取平均值\n")

    # ---- 按稀疏度分表 ----
    print("=" * 100)
    print("稀疏矩阵乘法性能测试")
    print("=" * 100)

    for sparsity in SPARSITIES:
        print(f"\n{'=' * 100}")
        print(f"稀疏度: {sparsity*100:.1f}%")
        print(f"{'=' * 100}")

        header = f"{'进程数':>8} | {'规模':>8} | {'时间(ms)':>10} | {'nnz':>10} | {'理论加速比':>10}"
        print(header)
        print("-" * len(header))

        results = {}

        for np in PROCESSES:
            if np > max_procs:
                print(f"{np:>8} | (跳过，超过核心数 {max_procs})")
                continue

            results[np] = {}

            for size in SIZES:
                print(f"  测试 np={np}, size={size}, sp={sparsity}", end="", flush=True)
                times = []
                nnzs = []
                speedups = []

                for run in range(RUNS):
                    t, nnz, speedup = run_test(np, size, sparsity)
                    if t is not None:
                        times.append(t)
                        if nnz is not None:
                            nnzs.append(nnz)
                        if speedup is not None:
                            speedups.append(speedup)
                    print(".", end="", flush=True)

                if times:
                    avg_time = sum(times) / len(times)
                    avg_nnz = sum(nnzs) / len(nnzs) if nnzs else 0
                    avg_speedup = sum(speedups) / len(speedups) if speedups else 0

                    results[np][size] = avg_time
                    print(f" -> {avg_time:.2f} ms")

                    row = f"{np:>8} | {size:>8} | {avg_time:>9.2f}ms | {avg_nnz:>9.0f} | {avg_speedup:>9.1f}x"
                    print(row)
                else:
                    results[np][size] = None
                    print(" -> 失败")

        # ---- 加速比分析 ----
        print(f"\n加速比分析（以 1 进程为基准，稀疏度={sparsity*100:.1f}%）")
        print("-" * 52)

        for size in SIZES:
            base_time = results.get(1, {}).get(size)
            if base_time is None or base_time == 0:
                continue

            print(f"\n矩阵规模: {size}x{size}")
            print(f"{'进程数':>8} | {'时间(ms)':>10} | {'加速比':>8} | {'效率':>8}")
            print("-" * 42)

            for np in PROCESSES:
                if np > max_procs:
                    continue
                t = results.get(np, {}).get(size)
                if t is not None and t > 0:
                    speedup = base_time / t
                    efficiency = speedup / np * 100
                    print(f"{np:>8} | {t:>9.2f}ms | {speedup:>7.2f}x | {efficiency:>7.1f}%")
                else:
                    print(f"{np:>8} | N/A")

    # ---- 不同稀疏度对比 ----
    print("\n" + "=" * 100)
    print("稀疏度对比（固定进程数=4，不同稀疏度下的性能）")
    print("=" * 100)

    for size in SIZES:
        print(f"\n矩阵规模: {size}x{size}")
        header = f"{'稀疏度':>8} | {'1p(ms)':>10} | {'2p(ms)':>10} | {'4p(ms)':>10} | {'nnz':>10} | {'理论加速比':>12}"
        print(header)
        print("-" * len(header))

        for sparsity in SPARSITIES:
            t1 = None
            t2 = None
            t4 = None
            nnz_avg = 0
            speedup_avg = 0

            for _ in range(RUNS):
                r, n, s = run_test(1, size, sparsity)
                if r is not None:
                    t1 = r if t1 is None else t1  # 取一次即可
                    break

            for _ in range(RUNS):
                r, n, s = run_test(2, size, sparsity)
                if r is not None:
                    t2 = r if t2 is None else t2
                    break

            for _ in range(RUNS):
                r, n, s = run_test(4, size, sparsity)
                if r is not None:
                    t4 = r if t4 is None else t4
                    if n is not None:
                        nnz_avg = n
                    if s is not None:
                        speedup_avg = s
                    break

            t1_str = f"{t1:>9.2f}ms" if t1 is not None else "N/A"
            t2_str = f"{t2:>9.2f}ms" if t2 is not None else "N/A"
            t4_str = f"{t4:>9.2f}ms" if t4 is not None else "N/A"
            nnz_str = f"{nnz_avg:>9.0f}" if nnz_avg > 0 else "N/A"
            speedup_str = f"{speedup_avg:>11.1f}x" if speedup_avg > 0 else "N/A"

            print(f"{sparsity*100:>7.1f}% | {t1_str} | {t2_str} | {t4_str} | {nnz_str} | {speedup_str}")

if __name__ == "__main__":
    main()
