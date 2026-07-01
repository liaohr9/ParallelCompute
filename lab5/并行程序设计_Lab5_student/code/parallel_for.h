#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void *(*parallel_functor)(int idx, void *arg);

int parallel_for(int start, int end, int inc,
                 parallel_functor functor, void *arg, int num_threads);

#ifdef __cplusplus
}
#endif

#endif
