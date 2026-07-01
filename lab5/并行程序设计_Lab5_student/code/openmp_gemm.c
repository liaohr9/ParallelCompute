#include "gemm_common.h"

#include <omp.h>
#include <string.h>

typedef enum {
    SCHED_DEFAULT,
    SCHED_STATIC_1,
    SCHED_DYNAMIC_1
} schedule_kind;

static schedule_kind parse_schedule(const char *name) {
    if (strcmp(name, "default") == 0) {
        return SCHED_DEFAULT;
    }
    if (strcmp(name, "static1") == 0 || strcmp(name, "static") == 0) {
        return SCHED_STATIC_1;
    }
    if (strcmp(name, "dynamic1") == 0 || strcmp(name, "dynamic") == 0) {
        return SCHED_DYNAMIC_1;
    }
    fprintf(stderr, "unknown schedule: %s\n", name);
    exit(2);
}

static const char *schedule_name(schedule_kind sched) {
    switch (sched) {
    case SCHED_DEFAULT:
        return "default";
    case SCHED_STATIC_1:
        return "static1";
    case SCHED_DYNAMIC_1:
        return "dynamic1";
    }
    return "unknown";
}

static void gemm_openmp(const double *a, const double *bt, double *c,
                        int m, int n, int k, int threads, schedule_kind sched) {
    if (sched == SCHED_DEFAULT) {
#pragma omp parallel for num_threads(threads)
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
    } else if (sched == SCHED_STATIC_1) {
#pragma omp parallel for num_threads(threads) schedule(static, 1)
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
    } else {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
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
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s M N K threads schedule(default|static1|dynamic1) [--print]\n",
            argv0);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        usage(argv[0]);
        return 2;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int threads = atoi(argv[4]);
    schedule_kind sched = parse_schedule(argv[5]);
    int do_print = argc > 6 && strcmp(argv[6], "--print") == 0;

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

    double t0 = wall_seconds();
    gemm_openmp(a, bt, c, m, n, k, threads, sched);
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
    printf("openmp,%s,%d,%d,%d,%d,%.6f,%.3f,%.10e,%.3e\n",
           schedule_name(sched), m, n, k, threads, elapsed, gflops,
           checksum_matrix(c, (size_t)m, (size_t)k), error);

    free(a);
    free(b);
    free(bt);
    free(c);
    free(ref);
    return 0;
}
