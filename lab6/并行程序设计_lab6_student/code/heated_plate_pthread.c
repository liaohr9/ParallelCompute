#define _POSIX_C_SOURCE 200809L

#include "parallel_for.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef M
#define M 500
#endif

#ifndef N
#define N 500
#endif

#define EPSILON 0.001
#define ALIGNMENT 64

typedef struct {
    double *u;
    double *w;
    double *diffs;
    int n;
} plate_args_t;

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void *xaligned_alloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

static void init_rows(int begin, int end, int tid, void *raw) {
    (void)tid;
    plate_args_t *a = (plate_args_t *)raw;
    for (int i = begin; i < end; ++i) {
        double *urow = a->u + (size_t)i * a->n;
        double *wrow = a->w + (size_t)i * a->n;
        for (int j = 1; j < a->n - 1; ++j) {
            urow[j] = 75.0;
            wrow[j] = 75.0;
        }
    }
}

static void update_rows(int begin, int end, int tid, void *raw) {
    plate_args_t *a = (plate_args_t *)raw;
    const int n = a->n;
    const double *restrict u = a->u;
    double *restrict w = a->w;
    double local_diff = 0.0;

    for (int i = begin; i < end; ++i) {
        const double *restrict north = u + (size_t)(i - 1) * n;
        const double *restrict curr = u + (size_t)i * n;
        const double *restrict south = u + (size_t)(i + 1) * n;
        double *restrict out = w + (size_t)i * n;

        for (int j = 1; j < n - 1; ++j) {
            double v = 0.25 * (north[j] + south[j] + curr[j - 1] + curr[j + 1]);
            double d = fabs(v - curr[j]);
            out[j] = v;
            if (d > local_diff) {
                local_diff = d;
            }
        }
    }

    double *thread_diff = &a->diffs[(size_t)tid * 8];
    if (local_diff > *thread_diff) {
        *thread_diff = local_diff;
    }
}

static void set_boundary(double *a, int m, int n) {
    for (int i = 1; i < m - 1; ++i) {
        a[(size_t)i * n] = 100.0;
        a[(size_t)i * n + n - 1] = 100.0;
    }
    for (int j = 0; j < n; ++j) {
        a[(size_t)(m - 1) * n + j] = 100.0;
        a[j] = 0.0;
    }
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    pf_schedule_t schedule = argc > 2 ? pf_parse_schedule(argv[2]) : PF_STATIC;
    int chunk = argc > 3 ? atoi(argv[3]) : 8;
    if (threads <= 0) {
        threads = 1;
    }
    if (chunk <= 0) {
        chunk = 8;
    }

    size_t bytes = (size_t)M * N * sizeof(double);
    double *u = (double *)xaligned_alloc(ALIGNMENT, bytes);
    double *w = (double *)xaligned_alloc(ALIGNMENT, bytes);
    double *diffs = (double *)xaligned_alloc(ALIGNMENT, (size_t)threads * 8 * sizeof(double));
    if (!u || !w || !diffs) {
        fprintf(stderr, "allocation failed\n");
        free(u);
        free(w);
        free(diffs);
        return 1;
    }
    memset(u, 0, bytes);
    memset(w, 0, bytes);
    memset(diffs, 0, (size_t)threads * 8 * sizeof(double));

    set_boundary(u, M, N);
    set_boundary(w, M, N);

    parallel_for_t *pf = parallel_for_create(threads, schedule, chunk);
    if (!pf) {
        fprintf(stderr, "parallel_for_create failed\n");
        free(u);
        free(w);
        free(diffs);
        return 1;
    }

    plate_args_t args = {.u = u, .w = w, .diffs = diffs, .n = N};
    parallel_for_run(pf, 1, M - 1, init_rows, &args);

    double start = wall_seconds();
    int iterations = 0;
    double diff = EPSILON;
    while (diff >= EPSILON) {
        memset(diffs, 0, (size_t)threads * 8 * sizeof(double));
        args.u = u;
        args.w = w;
        parallel_for_run(pf, 1, M - 1, update_rows, &args);

        diff = 0.0;
        for (int t = 0; t < threads; ++t) {
            double d = diffs[(size_t)t * 8];
            if (d > diff) {
                diff = d;
            }
        }

        double *tmp = u;
        u = w;
        w = tmp;
        iterations++;
    }
    double elapsed = wall_seconds() - start;

    double checksum = 0.0;
    for (int i = 0; i < M; i += 31) {
        for (int j = 0; j < N; j += 31) {
            checksum += u[(size_t)i * N + j];
        }
    }

    printf("impl,schedule,M,N,threads,chunk,iterations,diff,time_sec,checksum\n");
    printf("pthread,%s,%d,%d,%d,%d,%d,%.9f,%.6f,%.6f\n",
           pf_schedule_name(schedule), M, N, threads, chunk,
           iterations, diff, elapsed, checksum);

    parallel_for_destroy(pf);
    free(u);
    free(w);
    free(diffs);
    return 0;
}
