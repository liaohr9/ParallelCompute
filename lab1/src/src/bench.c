#include "common.h"
#include "kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// qsort 需要的比较器，用于把多次测量结果排序后取中位数。
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

// 通过命令行参数选择具体 kernel，这样一个 benchmark 驱动程序
// 就可以复用到基线版本、循环顺序优化版本和循环展开版本。
static matmul_kernel_t select_kernel(const char *name) {
    if (strcmp(name, "ijk") == 0) {
        return matmul_ijk;
    }
    if (strcmp(name, "ikj") == 0) {
        return matmul_ikj;
    }
    if (strcmp(name, "unroll4") == 0) {
        return matmul_unroll4;
    }
    return NULL;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <kernel> <n> <repeat>\n", prog);
    fprintf(stderr, "  kernel: ijk | ikj | unroll4\n");
}

int main(int argc, char **argv) {
    if (argc != 4) {
        usage(argv[0]);
        return 1;
    }

    const char *kernel_name = argv[1];
    size_t n = (size_t)strtoull(argv[2], NULL, 10);
    int repeat = atoi(argv[3]);
    if (n == 0 || repeat <= 0) {
        usage(argv[0]);
        return 1;
    }

    matmul_kernel_t kernel = select_kernel(kernel_name);
    if (kernel == NULL) {
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

    // 先做一次预热，避免首次运行时的缓存缺失、页错误等一次性开销
    // 直接污染正式计时结果。
    zero_matrix(c, n);
    kernel(a, b, c, n);

    for (int r = 0; r < repeat; ++r) {
        // ikj 和 unroll4 采用累加写法，因此每次测量前都要把结果矩阵清零。
        zero_matrix(c, n);
        double start = now_seconds();
        kernel(a, b, c, n);
        double end = now_seconds();
        times[r] = end - start;
    }

    qsort(times, (size_t)repeat, sizeof(double), compare_doubles);
    double median = times[repeat / 2];

    // 输出 checksum 供脚本做正确性校验。
    // 用总和而不是整矩阵回传，既轻量又足以快速发现大多数实现错误。
    double checksum = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        checksum += c[i];
    }

    // 输出固定的 key=value 格式，便于 Python 脚本统一解析各版本结果。
    printf("kernel=%s\n", kernel_name);
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
