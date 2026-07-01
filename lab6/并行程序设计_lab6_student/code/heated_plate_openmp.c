#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M
#define M 500
#endif

#ifndef N
#define N 500
#endif

#define EPSILON 0.001
#define ALIGNMENT 64

static void *xaligned_alloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

static void set_boundary(double *a) {
    for (int i = 1; i < M - 1; ++i) {
        a[(size_t)i * N] = 100.0;
        a[(size_t)i * N + N - 1] = 100.0;
    }
    for (int j = 0; j < N; ++j) {
        a[(size_t)(M - 1) * N + j] = 100.0;
        a[j] = 0.0;
    }
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : omp_get_max_threads();
    const char *schedule = argc > 2 ? argv[2] : "static";
    int chunk = argc > 3 ? atoi(argv[3]) : 8;
    if (threads <= 0) {
        threads = 1;
    }
    if (chunk <= 0) {
        chunk = 8;
    }
    omp_set_num_threads(threads);
    if (strcmp(schedule, "dynamic") == 0) {
        omp_set_schedule(omp_sched_dynamic, chunk);
    } else if (strcmp(schedule, "guided") == 0) {
        omp_set_schedule(omp_sched_guided, chunk);
    } else {
        omp_set_schedule(omp_sched_static, chunk);
        schedule = "static";
    }

    size_t bytes = (size_t)M * N * sizeof(double);
    double *u = (double *)xaligned_alloc(ALIGNMENT, bytes);
    double *w = (double *)xaligned_alloc(ALIGNMENT, bytes);
    if (!u || !w) {
        fprintf(stderr, "allocation failed\n");
        free(u);
        free(w);
        return 1;
    }
    memset(u, 0, bytes);
    memset(w, 0, bytes);
    set_boundary(u);
    set_boundary(w);

#pragma omp parallel for schedule(static)
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            u[(size_t)i * N + j] = 75.0;
            w[(size_t)i * N + j] = 75.0;
        }
    }

    double start = omp_get_wtime();
    int iterations = 0;
    double diff = EPSILON;
    while (diff >= EPSILON) {
        diff = 0.0;
#pragma omp parallel for schedule(runtime) reduction(max : diff)
        for (int i = 1; i < M - 1; ++i) {
            const double *restrict north = u + (size_t)(i - 1) * N;
            const double *restrict curr = u + (size_t)i * N;
            const double *restrict south = u + (size_t)(i + 1) * N;
            double *restrict out = w + (size_t)i * N;
            for (int j = 1; j < N - 1; ++j) {
                double v = 0.25 * (north[j] + south[j] + curr[j - 1] + curr[j + 1]);
                double d = fabs(v - curr[j]);
                out[j] = v;
                if (d > diff) {
                    diff = d;
                }
            }
        }
        double *tmp = u;
        u = w;
        w = tmp;
        iterations++;
    }
    double elapsed = omp_get_wtime() - start;

    double checksum = 0.0;
    for (int i = 0; i < M; i += 31) {
        for (int j = 0; j < N; j += 31) {
            checksum += u[(size_t)i * N + j];
        }
    }

    printf("impl,schedule,M,N,threads,chunk,iterations,diff,time_sec,checksum\n");
    printf("openmp,%s,%d,%d,%d,%d,%d,%.9f,%.6f,%.6f\n",
           schedule, M, N, threads, chunk, iterations, diff, elapsed, checksum);

    free(u);
    free(w);
    return 0;
}
