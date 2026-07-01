#define _POSIX_C_SOURCE 200809L
#include "common.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 为矩阵分配 64 字节对齐的连续内存，便于编译器向量化，
// 也与 MKL 常见的对齐习惯保持一致，减少不同版本之间的额外变量。
double *alloc_matrix(size_t n) {
    void *ptr = NULL;
    size_t bytes = n * n * sizeof(double);
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        return NULL;
    }
    return (double *)ptr;
}

void free_matrix(double *ptr) {
    free(ptr);
}

// 采用固定公式初始化矩阵，而不是使用随机数，
// 这样每次运行、每个版本都会得到完全一致的输入，便于做公平对比和校验。
void fill_matrix(double *mat, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            mat[i * n + j] = ((double)((i * 1315423911u + j * 2654435761u) % 1024) / 1024.0) - 0.5;
        }
    }
}

// C 矩阵在 ikj/unroll4 等实现里会被累加写入，因此每次计时前都必须先清零。
// 这里直接用 memset 清整块内存，开销低且代码简单。
void zero_matrix(double *mat, size_t n) {
    memset(mat, 0, n * n * sizeof(double));
}

// 使用单调时钟计时，避免系统时间被校时或手工修改时影响 benchmark 结果。
double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// 保留一个逐元素误差工具函数，便于后续需要时比较两个结果矩阵是否一致。
double max_abs_diff(const double *a, const double *b, size_t n) {
    double max_diff = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        double diff = fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

// 经典矩阵乘法的浮点操作数近似为 2 * N^3：
// 每个乘加对记作 2 次浮点运算，据此换算为 GFLOPS。
double compute_gflops(size_t n, double seconds) {
    if (seconds <= 0.0) {
        return 0.0;
    }
    double flops = 2.0 * (double)n * (double)n * (double)n;
    return flops / seconds / 1e9;
}
