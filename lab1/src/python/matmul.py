#!/usr/bin/env python3
import statistics
import sys
import time


# 与 C 版本使用相同的确定性初始化公式，保证不同语言实现拿到同一组输入数据。
def fill_matrix(n: int):
    return [[((i * 1315423911 + j * 2654435761) % 1024) / 1024.0 - 0.5 for j in range(n)] for i in range(n)]


# Python 基线版本直接构造全新的零矩阵，逻辑最直观，也便于和 C 版本对齐。
def zero_matrix(n: int):
    return [[0.0 for _ in range(n)] for _ in range(n)]


# 最朴素的 ijk 三重循环实现，作为 1 Python 版本的对照基线。
def matmul_ijk(a, b, c, n: int):
    for i in range(n):
        row_a = a[i]
        row_c = c[i]
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += row_a[k] * b[k][j]
            row_c[j] = s


# 与 C 端保持一致，统一用 2 * N^3 估算浮点操作次数。
def gflops(n: int, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return (2.0 * n * n * n) / seconds / 1e9


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <n> <repeat>", file=sys.stderr)
        return 1

    n = int(sys.argv[1])
    repeat = int(sys.argv[2])
    if n <= 0 or repeat <= 0:
        return 1

    a = fill_matrix(n)
    b = fill_matrix(n)
    c = zero_matrix(n)

    # 预热一次，尽量把解释器启动后首次执行的额外扰动移出正式计时。
    matmul_ijk(a, b, c, n)

    times = []
    for _ in range(repeat):
        c = zero_matrix(n)
        start = time.perf_counter()
        matmul_ijk(a, b, c, n)
        end = time.perf_counter()
        times.append(end - start)

    # 采用中位数而非单次结果，降低 Python 运行时抖动对结论的影响。
    median = statistics.median(times)
    checksum = sum(sum(row) for row in c)

    # 输出格式与 C 版本统一，便于同一个脚本解析。
    print("kernel=python")
    print(f"n={n}")
    print(f"repeat={repeat}")
    print(f"median_sec={median:.9f}")
    print(f"gflops={gflops(n, median):.6f}")
    print(f"checksum={checksum:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
