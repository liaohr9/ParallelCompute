#include "parallel_for.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    parallel_for_t *pf;
    int tid;
} worker_arg_t;

struct parallel_for {
    int num_threads;
    pf_schedule_t schedule;
    int chunk_size;

    pthread_t *threads;
    worker_arg_t *worker_args;

    pthread_mutex_t mutex;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;

    int generation;
    int active_workers;
    int stop;

    int start;
    int end;
    pf_body_t body;
    void *arg;
    atomic_int next;
};

static int min_int(int a, int b) {
    return a < b ? a : b;
}

static int max_int(int a, int b) {
    return a > b ? a : b;
}

static int claim_work(parallel_for_t *pf, int *begin, int *end) {
    const int total_end = pf->end;
    int b;
    int chunk;

    if (pf->schedule == PF_DYNAMIC) {
        chunk = pf->chunk_size;
        b = atomic_fetch_add_explicit(&pf->next, chunk, memory_order_relaxed);
    } else {
        for (;;) {
            b = atomic_load_explicit(&pf->next, memory_order_relaxed);
            if (b >= total_end) {
                return 0;
            }
            int remaining = total_end - b;
            chunk = max_int(pf->chunk_size, remaining / (pf->num_threads * 2));
            if (atomic_compare_exchange_weak_explicit(
                    &pf->next, &b, b + chunk,
                    memory_order_relaxed, memory_order_relaxed)) {
                break;
            }
        }
    }

    if (b >= total_end) {
        return 0;
    }
    *begin = b;
    *end = min_int(b + chunk, total_end);
    return 1;
}

static void run_static(parallel_for_t *pf, int tid) {
    int total = pf->end - pf->start;
    int base = total / pf->num_threads;
    int extra = total % pf->num_threads;
    int offset = tid * base + min_int(tid, extra);
    int count = base + (tid < extra);
    if (count > 0) {
        pf->body(pf->start + offset, pf->start + offset + count, tid, pf->arg);
    }
}

static void run_dynamic_or_guided(parallel_for_t *pf, int tid) {
    int begin;
    int end;
    while (claim_work(pf, &begin, &end)) {
        pf->body(begin, end, tid, pf->arg);
    }
}

static void *worker_main(void *raw) {
    worker_arg_t *wa = (worker_arg_t *)raw;
    parallel_for_t *pf = wa->pf;
    int tid = wa->tid;
    int seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&pf->mutex);
        while (!pf->stop && pf->generation == seen_generation) {
            pthread_cond_wait(&pf->start_cond, &pf->mutex);
        }
        if (pf->stop) {
            pthread_mutex_unlock(&pf->mutex);
            return NULL;
        }
        seen_generation = pf->generation;
        pthread_mutex_unlock(&pf->mutex);

        if (pf->schedule == PF_STATIC) {
            run_static(pf, tid);
        } else {
            run_dynamic_or_guided(pf, tid);
        }

        pthread_mutex_lock(&pf->mutex);
        pf->active_workers--;
        if (pf->active_workers == 0) {
            pthread_cond_signal(&pf->done_cond);
        }
        pthread_mutex_unlock(&pf->mutex);
    }
}

parallel_for_t *parallel_for_create(int num_threads, pf_schedule_t schedule, int chunk_size) {
    if (num_threads <= 0) {
        return NULL;
    }
    if (chunk_size <= 0) {
        chunk_size = 1;
    }

    parallel_for_t *pf = (parallel_for_t *)calloc(1, sizeof(*pf));
    if (!pf) {
        return NULL;
    }
    pf->num_threads = num_threads;
    pf->schedule = schedule;
    pf->chunk_size = chunk_size;
    atomic_init(&pf->next, 0);

    pf->threads = (pthread_t *)calloc((size_t)num_threads, sizeof(*pf->threads));
    pf->worker_args = (worker_arg_t *)calloc((size_t)num_threads, sizeof(*pf->worker_args));
    if (!pf->threads || !pf->worker_args) {
        parallel_for_destroy(pf);
        return NULL;
    }

    pthread_mutex_init(&pf->mutex, NULL);
    pthread_cond_init(&pf->start_cond, NULL);
    pthread_cond_init(&pf->done_cond, NULL);

    for (int t = 0; t < num_threads; ++t) {
        pf->worker_args[t].pf = pf;
        pf->worker_args[t].tid = t;
        if (pthread_create(&pf->threads[t], NULL, worker_main, &pf->worker_args[t]) != 0) {
            pf->num_threads = t;
            parallel_for_destroy(pf);
            return NULL;
        }
    }

    return pf;
}

int parallel_for_run(parallel_for_t *pf, int start, int end, pf_body_t body, void *arg) {
    if (!pf || !body || end < start) {
        return -1;
    }
    if (end == start) {
        return 0;
    }

    pthread_mutex_lock(&pf->mutex);
    pf->start = start;
    pf->end = end;
    pf->body = body;
    pf->arg = arg;
    pf->active_workers = pf->num_threads;
    atomic_store_explicit(&pf->next, start, memory_order_relaxed);
    pf->generation++;
    pthread_cond_broadcast(&pf->start_cond);
    while (pf->active_workers > 0) {
        pthread_cond_wait(&pf->done_cond, &pf->mutex);
    }
    pthread_mutex_unlock(&pf->mutex);
    return 0;
}

void parallel_for_destroy(parallel_for_t *pf) {
    if (!pf) {
        return;
    }
    if (pf->threads) {
        pthread_mutex_lock(&pf->mutex);
        pf->stop = 1;
        pthread_cond_broadcast(&pf->start_cond);
        pthread_mutex_unlock(&pf->mutex);
        for (int t = 0; t < pf->num_threads; ++t) {
            pthread_join(pf->threads[t], NULL);
        }
    }
    pthread_cond_destroy(&pf->done_cond);
    pthread_cond_destroy(&pf->start_cond);
    pthread_mutex_destroy(&pf->mutex);
    free(pf->worker_args);
    free(pf->threads);
    free(pf);
}

pf_schedule_t pf_parse_schedule(const char *name) {
    if (!name || strcmp(name, "static") == 0) {
        return PF_STATIC;
    }
    if (strcmp(name, "dynamic") == 0) {
        return PF_DYNAMIC;
    }
    if (strcmp(name, "guided") == 0) {
        return PF_GUIDED;
    }
    return PF_STATIC;
}

const char *pf_schedule_name(pf_schedule_t schedule) {
    switch (schedule) {
    case PF_DYNAMIC:
        return "dynamic";
    case PF_GUIDED:
        return "guided";
    case PF_STATIC:
    default:
        return "static";
    }
}
