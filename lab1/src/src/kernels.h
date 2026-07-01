#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>

typedef void (*matmul_kernel_t)(const double *a, const double *b, double *c, size_t n);

void matmul_ijk(const double *a, const double *b, double *c, size_t n);
void matmul_ikj(const double *a, const double *b, double *c, size_t n);
void matmul_unroll4(const double *a, const double *b, double *c, size_t n);

#endif
