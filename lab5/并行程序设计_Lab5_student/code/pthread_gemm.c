#include "gemm_common.h"
#include "parallel_for.h"

#include <string.h>

typedef struct {
    const double *a;
    const double *bt;
    double *c;
    int n;
    int k;
} gemm_args;

static void *gemm_row(int i, void *raw) {
    gemm_args *args = (gemm_args *)raw;
    for (int j = 0; j < args->k; ++j) {
        double sum = 0.0;
        const double *arow = args->a + (size_t)i * args->n;
        const double *brow = args->bt + (size_t)j * args->n;
        for (int p = 0; p < args->n; ++p) {
            sum += arow[p] * brow[p];
        }
        args->c[(size_t)i * args->k + j] = sum;
    }
    return NULL;
}

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s M N K threads [--print]\n", argv0);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        usage(argv[0]);
        return 2;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int threads = atoi(argv[4]);
    int do_print = argc > 5 && strcmp(argv[5], "--print") == 0;
    if (m <= 0 || n <= 0 || k <= 0 || threads <= 0) {
        usage(argv[0]);
        return 2;
    }

    double *a = alloc_matrix((size_t)m, (size_t)n);
    double *b = alloc_matrix((size_t)n, (size_t)k);
    double *bt = alloc_matrix((size_t)k, (size_t)n);
    double *c = alloc_matrix((size_t)m, (size_t)k);
    double *ref = NULL;
    if (!a || !b || !bt || !c) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    fill_matrix(a, (size_t)m, (size_t)n, 17u);
    fill_matrix(b, (size_t)n, (size_t)k, 29u);
    transpose_matrix(b, bt, n, k);
    zero_matrix(c, (size_t)m, (size_t)k);

    gemm_args args = {a, bt, c, n, k};
    double t0 = wall_seconds();
    if (parallel_for(0, m, 1, gemm_row, &args, threads) != 0) {
        fprintf(stderr, "parallel_for failed\n");
        return 1;
    }
    double elapsed = wall_seconds() - t0;

    double error = -1.0;
    if (m <= 256 && n <= 256 && k <= 256) {
        ref = alloc_matrix((size_t)m, (size_t)k);
        if (!ref) {
            fprintf(stderr, "reference allocation failed\n");
            return 1;
        }
        gemm_serial(a, bt, ref, m, n, k);
        error = max_abs_diff(c, ref, (size_t)m, (size_t)k);
    }

    if (do_print) {
        print_matrix("A", a, m, n);
        print_matrix("B", b, n, k);
        print_matrix("C", c, m, k);
    }

    double gflops = 2.0 * (double)m * (double)n * (double)k / elapsed / 1e9;
    printf("impl,schedule,M,N,K,threads,time_sec,gflops,checksum,max_error\n");
    printf("pthread,static-block,%d,%d,%d,%d,%.6f,%.3f,%.10e,%.3e\n",
           m, n, k, threads, elapsed, gflops,
           checksum_matrix(c, (size_t)m, (size_t)k), error);

    free(a);
    free(b);
    free(bt);
    free(c);
    free(ref);
    return 0;
}
