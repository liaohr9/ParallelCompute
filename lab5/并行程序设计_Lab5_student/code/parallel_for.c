#include "parallel_for.h"

#include <pthread.h>
#include <stdlib.h>

typedef struct {
    int start;
    int iterations;
    int inc;
    parallel_functor functor;
    void *arg;
} worker_args;

static void *worker_main(void *raw) {
    worker_args *args = (worker_args *)raw;
    for (int t = 0; t < args->iterations; ++t) {
        int idx = args->start + t * args->inc;
        args->functor(idx, args->arg);
    }
    return NULL;
}

static int iteration_count(int start, int end, int inc) {
    if (inc == 0) {
        return -1;
    }
    if ((inc > 0 && start >= end) || (inc < 0 && start <= end)) {
        return 0;
    }
    int diff = inc > 0 ? end - start : start - end;
    int step = inc > 0 ? inc : -inc;
    return (diff + step - 1) / step;
}

int parallel_for(int start, int end, int inc,
                 parallel_functor functor, void *arg, int num_threads) {
    if (!functor || num_threads <= 0) {
        return -1;
    }

    int total = iteration_count(start, end, inc);
    if (total < 0) {
        return -1;
    }
    if (total == 0) {
        return 0;
    }
    if (num_threads > total) {
        num_threads = total;
    }

    pthread_t *threads = (pthread_t *)calloc((size_t)num_threads, sizeof(pthread_t));
    worker_args *args = (worker_args *)calloc((size_t)num_threads, sizeof(worker_args));
    if (!threads || !args) {
        free(threads);
        free(args);
        return -1;
    }

    int base = total / num_threads;
    int extra = total % num_threads;
    int offset = 0;
    int created = 0;
    for (int t = 0; t < num_threads; ++t) {
        int count = base + (t < extra ? 1 : 0);
        args[t].start = start + offset * inc;
        args[t].iterations = count;
        args[t].inc = inc;
        args[t].functor = functor;
        args[t].arg = arg;
        offset += count;
        if (pthread_create(&threads[t], NULL, worker_main, &args[t]) != 0) {
            created = t;
            for (int j = 0; j < created; ++j) {
                pthread_join(threads[j], NULL);
            }
            free(threads);
            free(args);
            return -1;
        }
        created = t + 1;
    }

    for (int t = 0; t < created; ++t) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(args);
    return 0;
}
