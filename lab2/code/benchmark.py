#!/usr/bin/env python3
"""
性能测试脚本：自动化运行不同进程数和矩阵规模的组合

用途：
  自动遍历所有进程数和矩阵规模的组合，收集每次运行的时间开销。

使用方法：
  chmod +x benchmark.py
  ./benchmark.py
"""

import subprocess
import sys
import os

# ---- 测试参数 ----
PROCESSES = [1, 2, 4, 8, 16, 32]       # 进程数
SIZES = [128, 256, 512, 1024, 2048, 4096] # 矩阵规模
RUNS = 10  # 每组参数重复运行次数，取平均值以减小波动

PROGRAM = "./mpi_matrix"

def check_program():
    """检查程序是否已编译"""
    if not os.path.exists(PROGRAM):
        print("程序未编译，正在执行 make...")
        result = subprocess.run(["make"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"编译失败：\n{result.stderr}")
            sys.exit(1)
        print("编译完成。")

def run_test(np, size):
    """运行一次测试，返回时间（秒），失败返回 None"""
    try:
        result = subprocess.run(
            ["mpirun", "-np", str(np), "--oversubscribe", PROGRAM, str(size), str(size), str(size)],
            capture_output=True,
            text=True,
            timeout=36000  # 600 分钟超时，大矩阵计算耗时较长
        )
        # 解析输出中的"计算时间: X.XXXXXX 秒"
        for line in result.stdout.split('\n'):
            if '计算时间:' in line:
                # 提取数值部分
                parts = line.split()
                for i, part in enumerate(parts):
                    if '计算时间:' in part and i + 1 < len(parts):
                        return float(parts[i + 1])
        return None
    except Exception as e:
        print(f"  运行失败 (np={np}, size={size}): {e}")
        return None

def main():
    check_program()

    # 获取本机逻辑核心数
    try:
        import multiprocessing
        max_procs = multiprocessing.cpu_count()
    except:
        max_procs = 8

    print(f"本机 CPU 逻辑核心数: {max_procs}")
    print(f"每组参数运行 {RUNS} 次取平均值")
    print()

    # 打印表头
    header = f"{'进程数':>8}"
    for size in SIZES:
        header += f" | {size:>8}"
    print(header)
    print("-" * len(header))

    # 运行测试
    results = {}

    for np in PROCESSES:
        if np > max_procs:
            print(f"{np:>8} | (跳过，超过核心数 {max_procs})")
            continue

        row = f"{np:>8}"
        results[np] = {}

        for size in SIZES:
            print(f"\n正在测试: np={np}, size={size}x{size}", end="")
            times = []
            for run in range(RUNS):
                t = run_test(np, size)
                if t is not None:
                    times.append(t)
                print(".", end="", flush=True)

            if times:
                avg = sum(times) / len(times)
                results[np][size] = avg
                # 将秒转换为毫秒显示
                avg_ms = avg * 1000
                row += f" | {avg_ms:>7.2f}ms"
            else:
                results[np][size] = None
                row += " | N/A     "

            print(f" -> {avg_ms:.2f} ms" if times else " -> 失败")

        print(row)

    # ---- 输出加速比分析 ----
    print("\n" + "=" * 60)
    print("加速比分析（以 1 进程为基准）")
    print("=" * 60)

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
                print(f"{np:>8} | {t*1000:>9.2f}ms | {speedup:>7.2f}x | {efficiency:>7.1f}%")
            else:
                print(f"{np:>8} | N/A")

if __name__ == "__main__":
    main()
