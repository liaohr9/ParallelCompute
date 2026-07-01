CC := gcc
CFLAGS := -O3 -march=native -std=c11 -Wall -Wextra -D_POSIX_C_SOURCE=200809L
LDFLAGS :=

.PHONY: all clean

all: openmp_gemm libparallel_for.so pthread_gemm

openmp_gemm: openmp_gemm.c gemm_common.h
	$(CC) $(CFLAGS) -fopenmp openmp_gemm.c -o $@

parallel_for.o: parallel_for.c parallel_for.h
	$(CC) $(CFLAGS) -fPIC -c parallel_for.c -o $@

libparallel_for.so: parallel_for.o
	$(CC) -shared -pthread parallel_for.o -o $@

pthread_gemm: pthread_gemm.c gemm_common.h parallel_for.h libparallel_for.so
	$(CC) $(CFLAGS) pthread_gemm.c -L. -lparallel_for -Wl,-rpath,'$$ORIGIN' -pthread -o $@

clean:
	rm -f openmp_gemm pthread_gemm parallel_for.o libparallel_for.so
