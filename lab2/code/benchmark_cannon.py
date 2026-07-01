#!/usr/bin/env python3
"""
Cannon 算法性能测试脚本

遍历不同进程数和矩阵规模，对比 Cannon 算法 vs 原始方案的内存和性能。

使用方法：
    ./benchmark_cannon.py
"""

import subprocess
import sys
import os
import math

# ---- 测试参数 ----
PROCESSES = [1, 4, 9, 16, 32]          # 必须是完全平方数
SIZES = [128, 256, 512, 1024, 2048, 4096]  # 方阵大小
RUNS = 5  # 每组重复运行次数

PROGRAM = "./mpi_cannon"

def check_program():
    if not os.path.exists(PROGRAM):
        print("程序未编译，正在执行 make...")
        result = subprocess.run(["make", "mpi_cannon"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"编译失败：\n{result.stderr}")
            sys.exit(1)

def is_perfect_square(n):
    r = int(math.isqrt(n))
    return r * r == n

def run_test(np, size):
    """运行一次测试，返回 (time_ms, memory_per_process_mb, memory_original_mb)"""
    try:
        result = subprocess.run(
            ["mpirun", "-np", str(np), "--oversubscribe", PROGRAM, str(size)],
            capture_output=True, text=True, timeout=3600
        )
        time_ms = None
        memory_mb = None
        memory_orig = None
        for line in result.stdout.split('\n'):
            if '计算时间:' in line:
                # "计算时间: 0.002653 秒 (2.65 毫秒)"
                parts = line.split()
                for i, p in enumerate(parts):
                    if '计算时间:' in p and i + 3 < len(parts):
                        # parts[i+3] = "2.65", parts[i+4] = "毫秒)"
                        time_ms = float(parts[i + 3].lstrip('(').rstrip(')'))
                        break
            if '每进程内存:' in line:
                # "每进程内存: ~6.00 MB (原始方案需要 24.00 MB)"
                line_stripped = line.strip()
                try:
                    mb_part = line_stripped.split('~')[1].split(' ')[0]
                    memory_mb = float(mb_part)
                    orig_part = line_stripped.split('需要 ')[1].split(' ')[0]
                    memory_orig = float(orig_part)
                except:
                    pass
        return time_ms, memory_mb, memory_orig
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

    # ---- 性能表 ----
    print("=" * 90)
    print("Cannon 算法性能测试")
    print("=" * 90)

    header = f"{'进程数':>8} | {'规模':>8} | {'时间(ms)':>10} | {'本地内存':>10} | {'原始内存':>10} | {'内存节省':>10}"
    print(header)
    print("-" * len(header))

    results = {}

    for np in PROCESSES:
        if not is_perfect_square(np):
            continue
        if np > max_procs:
            print(f"{np:>8} | (跳过，超过核心数 {max_procs})")
            continue

        results[np] = {}

        for size in SIZES:
            print(f"  测试 np={np}, size={size}x{size}", end="", flush=True)
            times = []
            memories_mb = []
            memories_orig = []

            for run in range(RUNS):
                t, mem, mem_orig = run_test(np, size)
                if t is not None:
                    times.append(t)
                    if mem is not None:
                        memories_mb.append(mem)
                    if mem_orig is not None:
                        memories_orig.append(mem_orig)
                print(".", end="", flush=True)

            if times:
                avg_time = sum(times) / len(times)
                avg_mem = sum(memories_mb) / len(memories_mb) if memories_mb else 0
                avg_orig = sum(memories_orig) / len(memories_orig) if memories_orig else 0
                savings = (1 - avg_mem / avg_orig) * 100 if avg_orig > 0 else 0

                results[np][size] = avg_time
                mem_str = f"{avg_mem:.1f} MB" if avg_mem > 0 else "N/A"
                orig_str = f"{avg_orig:.1f} MB" if avg_orig > 0 else "N/A"
                print(f" -> {avg_time:.2f} ms, 内存 {mem_str}")

                row = f"{np:>8} | {size:>8} | {avg_time:>9.2f}ms | {mem_str:>10} | {orig_str:>10} | {savings:>8.0f}%"
                print(row)
            else:
                results[np][size] = None
                print(" -> 失败")

    # ---- 加速比分析 ----
    print("\n" + "=" * 90)
    print("加速比分析（以 1 进程为基准）")
    print("=" * 90)

    for size in SIZES:
        base_time = results.get(1, {}).get(size)
        if base_time is None or base_time == 0:
            continue

        print(f"\n矩阵规模: {size}x{size}")
        print(f"{'进程数':>8} | {'时间(ms)':>10} | {'加速比':>8} | {'效率':>8} | {'并行加速比':>10}")
        print("-" * 52)

        for np in PROCESSES:
            if not is_perfect_square(np):
                continue
            if np > max_procs:
                continue
            t = results.get(np, {}).get(size)
            if t is not None and t > 0:
                speedup = base_time / t
                efficiency = speedup / np * 100
                parallel_speedup = speedup
                print(f"{np:>8} | {t:>9.2f}ms | {speedup:>7.2f}x | {efficiency:>7.1f}% | {parallel_speedup:>9.2f}x")
            else:
                print(f"{np:>8} | N/A")

if __name__ == "__main__":
    main()
