#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>

double *alloc_matrix(size_t n);
void free_matrix(double *ptr);
void fill_matrix(double *mat, size_t n);
void zero_matrix(double *mat, size_t n);
double now_seconds(void);
double max_abs_diff(const double *a, const double *b, size_t n);
double compute_gflops(size_t n, double seconds);

#endif
