#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PF_STATIC = 0,
    PF_DYNAMIC = 1,
    PF_GUIDED = 2
} pf_schedule_t;

typedef void (*pf_body_t)(int begin, int end, int tid, void *arg);

typedef struct parallel_for parallel_for_t;

parallel_for_t *parallel_for_create(int num_threads, pf_schedule_t schedule, int chunk_size);
int parallel_for_run(parallel_for_t *pf, int start, int end, pf_body_t body, void *arg);
void parallel_for_destroy(parallel_for_t *pf);

pf_schedule_t pf_parse_schedule(const char *name);
const char *pf_schedule_name(pf_schedule_t schedule);

#ifdef __cplusplus
}
#endif

#endif
