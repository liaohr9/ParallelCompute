#include "kernels.h"

// 最基础的 ijk 三重循环版本，作为 C/C++ 基线实现。
// 这种访问顺序会频繁跨行读取 B，在缓存友好性上并不理想。
void matmul_ijk(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// 调整为 ikj 顺序后，最内层循环会顺序扫描 B 的一整行和 C 的一整行，
// 连续内存访问更多，更容易被缓存和硬件预取利用。
// 由于这里是累加写回 c_row[j]，调用前要求 C 已被清零。
void matmul_ikj(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            double aik = a[i * n + k];
            const double *b_row = b + k * n;
            double *c_row = c + i * n;
            for (size_t j = 0; j < n; ++j) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}

// 在 ikj 的基础上做 4 路循环展开，减少最内层循环的分支与索引开销，
// 也给编译器提供更直接的指令级并行机会。
void matmul_unroll4(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            double aik = a[i * n + k];
            const double *b_row = b + k * n;
            double *c_row = c + i * n;
            size_t j = 0;
            for (; j + 3 < n; j += 4) {
                c_row[j] += aik * b_row[j];
                c_row[j + 1] += aik * b_row[j + 1];
                c_row[j + 2] += aik * b_row[j + 2];
                c_row[j + 3] += aik * b_row[j + 3];
            }
            // 处理不能被 4 整除时剩下的尾部元素。
            for (; j < n; ++j) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}
