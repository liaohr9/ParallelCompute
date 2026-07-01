#include "common.h"

#include <stdio.h>
#include <stdlib.h>

// 直接声明 cblas_dgemm 所需的最小接口，
// 避免在不同安装环境下依赖特定的 MKL 头文件布局。
typedef int MKL_INT;

enum {
    CblasRowMajor = 101,
    CblasNoTrans = 111,
};

extern void cblas_dgemm(const int layout, const int transa, const int transb,
                        const MKL_INT m, const MKL_INT n, const MKL_INT k,
                        const double alpha, const double *a, const MKL_INT lda,
                        const double *b, const MKL_INT ldb, const double beta,
                        double *c, const MKL_INT ldc);

// 对多次测量结果排序，使用中位数作为最终运行时间，减少偶然抖动的影响。
static int compare_doubles(const void *lhs, const void *rhs) {
    double a = *(const double *)lhs;
    double b = *(const double *)rhs;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <n> <repeat>\n", prog);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        usage(argv[0]);
        return 1;
    }

    size_t n = (size_t)strtoull(argv[1], NULL, 10);
    int repeat = atoi(argv[2]);
    if (n == 0 || repeat <= 0) {
        usage(argv[0]);
        return 1;
    }

    double *a = alloc_matrix(n);
    double *b = alloc_matrix(n);
    double *c = alloc_matrix(n);
    double *times = (double *)malloc((size_t)repeat * sizeof(double));
    if (a == NULL || b == NULL || c == NULL || times == NULL) {
        fprintf(stderr, "allocation failed\n");
        free_matrix(a);
        free_matrix(b);
        free_matrix(c);
        free(times);
        return 2;
    }

    fill_matrix(a, n);
    fill_matrix(b, n);

    // 先执行一次预热，保证正式计时时已完成基础的内存分页和库加载。
    zero_matrix(c, n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (MKL_INT)n, (MKL_INT)n, (MKL_INT)n,
                1.0, a, (MKL_INT)n, b, (MKL_INT)n, 0.0, c, (MKL_INT)n);

    for (int r = 0; r < repeat; ++r) {
        zero_matrix(c, n);
        double start = now_seconds();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (MKL_INT)n, (MKL_INT)n, (MKL_INT)n,
                    1.0, a, (MKL_INT)n, b, (MKL_INT)n, 0.0, c, (MKL_INT)n);
        double end = now_seconds();
        times[r] = end - start;
    }

    qsort(times, (size_t)repeat, sizeof(double), compare_doubles);
    double median = times[repeat / 2];

    // 仍然输出 checksum，与其他版本沿用同一套正确性校验口径。
    double checksum = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        checksum += c[i];
    }

    printf("kernel=mkl\n");
    printf("n=%zu\n", n);
    printf("repeat=%d\n", repeat);
    printf("median_sec=%.9f\n", median);
    printf("gflops=%.6f\n", compute_gflops(n, median));
    printf("checksum=%.12f\n", checksum);

    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free(times);
    return 0;
}
