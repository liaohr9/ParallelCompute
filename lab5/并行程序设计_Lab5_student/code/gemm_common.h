#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline double *alloc_matrix(size_t rows, size_t cols) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, rows * cols * sizeof(double)) != 0) {
        return NULL;
    }
    return (double *)ptr;
}

static inline void fill_matrix(double *a, size_t rows, size_t cols, uint32_t seed) {
    uint32_t x = seed;
    for (size_t i = 0; i < rows * cols; ++i) {
        x = x * 1664525u + 1013904223u;
        a[i] = (double)(x & 0xffffu) / 65535.0;
    }
}

static inline void zero_matrix(double *a, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows * cols; ++i) {
        a[i] = 0.0;
    }
}

static inline void transpose_matrix(const double *b, double *bt, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            bt[(size_t)j * rows + i] = b[(size_t)i * cols + j];
        }
    }
}

static inline void gemm_serial(const double *a, const double *bt, double *c,
                               int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            const double *arow = a + (size_t)i * n;
            const double *brow = bt + (size_t)j * n;
            for (int p = 0; p < n; ++p) {
                sum += arow[p] * brow[p];
            }
            c[(size_t)i * k + j] = sum;
        }
    }
}

static inline double checksum_matrix(const double *a, size_t rows, size_t cols) {
    double s = 0.0;
    for (size_t i = 0; i < rows * cols; ++i) {
        s += a[i];
    }
    return s;
}

static inline double max_abs_diff(const double *a, const double *b, size_t rows, size_t cols) {
    double e = 0.0;
    for (size_t i = 0; i < rows * cols; ++i) {
        double d = fabs(a[i] - b[i]);
        if (d > e) {
            e = d;
        }
    }
    return e;
}

static inline void print_matrix(const char *name, const double *a, int rows, int cols) {
    printf("%s =\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.4f%c", a[(size_t)i * cols + j], j + 1 == cols ? '\n' : ' ');
        }
    }
}

#endif
