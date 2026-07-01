CC := gcc
CFLAGS := -O3 -march=native -mtune=native -std=c11 -Wall -Wextra -D_POSIX_C_SOURCE=200809L
LDFLAGS :=

.PHONY: all clean

all: heated_plate_pthread heated_plate_openmp

parallel_for.o: parallel_for.c parallel_for.h
	$(CC) $(CFLAGS) -c parallel_for.c -o $@

heated_plate_pthread: heated_plate_pthread.c parallel_for.o parallel_for.h
	$(CC) $(CFLAGS) heated_plate_pthread.c parallel_for.o -pthread -lm -o $@

heated_plate_openmp: heated_plate_openmp.c
	$(CC) $(CFLAGS) -fopenmp heated_plate_openmp.c -lm -o $@

clean:
	rm -f heated_plate_pthread heated_plate_openmp parallel_for.o
