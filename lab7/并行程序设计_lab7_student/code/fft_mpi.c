#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ccopy(int n, const double x[], double y[]);
static void cfft2_mpi(int n, double x[], double y[], const double w[], double sgn,
                      int rank, int size);
static void cffti(int n, double w[]);
static double ggl(double *ds);
static void partition_range(int total, int rank, int size, int *begin, int *end);
static void step_mpi(int n, int mj, const double in[], double out[], const double w[],
                     double sgn, int rank, int size);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ln2_max = argc > 1 ? atoi(argv[1]) : 20;
    int nits = argc > 2 ? atoi(argv[2]) : 10000;
    if (ln2_max < 1) {
        ln2_max = 1;
    }
    if (nits < 1) {
        nits = 1;
    }

    if (rank == 0) {
        printf("FFT_MPI\n");
        printf("  MPI C version\n");
        printf("  Processes = %d\n", size);
        printf("  Max ln2(N) = %d\n", ln2_max);
        printf("             N      NITS    Error         Time          Time/Call     MFLOPS\n");
    }

    double seed = 331.0;
    int n = 1;
    for (int ln2 = 1; ln2 <= ln2_max; ++ln2) {
        n *= 2;
        double *w = (double *)malloc((size_t)n * sizeof(double));
        double *x = (double *)malloc((size_t)2 * n * sizeof(double));
        double *y = (double *)malloc((size_t)2 * n * sizeof(double));
        double *z = (double *)malloc((size_t)2 * n * sizeof(double));
        if (!w || !x || !y || !z) {
            if (rank == 0) {
                fprintf(stderr, "allocation failed at n=%d\n", n);
            }
            free(w);
            free(x);
            free(y);
            free(z);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (rank == 0) {
            for (int i = 0; i < 2 * n; i += 2) {
                double z0 = ggl(&seed);
                double z1 = ggl(&seed);
                x[i] = z0;
                x[i + 1] = z1;
                z[i] = z0;
                z[i + 1] = z1;
            }
        }
        MPI_Bcast(x, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(z, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        cffti(n, w);

        cfft2_mpi(n, x, y, w, +1.0, rank, size);
        cfft2_mpi(n, y, x, w, -1.0, rank, size);

        double error = 0.0;
        if (rank == 0) {
            double fnm1 = 1.0 / (double)n;
            for (int i = 0; i < 2 * n; i += 2) {
                error += pow(z[i] - fnm1 * x[i], 2);
                error += pow(z[i + 1] - fnm1 * x[i + 1], 2);
            }
            error = sqrt(fnm1 * error);
            printf("  %12d  %8d  %12e", n, nits, error);
        }

        memset(x, 0, (size_t)2 * n * sizeof(double));
        memset(z, 0, (size_t)2 * n * sizeof(double));

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int it = 0; it < nits; ++it) {
            cfft2_mpi(n, x, y, w, +1.0, rank, size);
            cfft2_mpi(n, y, x, w, -1.0, rank, size);
        }
        double elapsed = MPI_Wtime() - t0;
        double max_elapsed = 0.0;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);
            double mflops = flops / 1.0E+06 / max_elapsed;
            printf("  %12e  %12e  %12f\n", max_elapsed,
                   max_elapsed / (double)(2 * nits), mflops);
        }

        if ((ln2 % 4) == 0) {
            nits /= 10;
            if (nits < 1) {
                nits = 1;
            }
        }

        free(w);
        free(x);
        free(y);
        free(z);
    }

    if (rank == 0) {
        printf("FFT_MPI: Normal end of execution.\n");
    }

    MPI_Finalize();
    return 0;
}

static void ccopy(int n, const double x[], double y[]) {
    memcpy(y, x, (size_t)2 * n * sizeof(double));
}

static void cfft2_mpi(int n, double x[], double y[], const double w[], double sgn,
                      int rank, int size) {
    int m = (int)(log((double)n) / log(1.99));
    int mj = 1;
    int tgle = 1;

    step_mpi(n, mj, x, y, w, sgn, rank, size);
    if (n == 2) {
        return;
    }

    for (int j = 0; j < m - 2; ++j) {
        mj *= 2;
        if (tgle) {
            step_mpi(n, mj, y, x, w, sgn, rank, size);
            tgle = 0;
        } else {
            step_mpi(n, mj, x, y, w, sgn, rank, size);
            tgle = 1;
        }
    }

    if (tgle) {
        ccopy(n, y, x);
    }

    mj = n / 2;
    step_mpi(n, mj, x, y, w, sgn, rank, size);
}

static void cffti(int n, double w[]) {
    const double pi = 3.141592653589793;
    int n2 = n / 2;
    double aw = 2.0 * pi / (double)n;
    for (int i = 0; i < n2; ++i) {
        double arg = aw * (double)i;
        w[i * 2 + 0] = cos(arg);
        w[i * 2 + 1] = sin(arg);
    }
}

static double ggl(double *seed) {
    double d2 = 0.2147483647e10;
    double t = fmod(16807.0 * *seed, d2);
    *seed = t;
    return (t - 1.0) / (d2 - 1.0);
}

static void partition_range(int total, int rank, int size, int *begin, int *end) {
    int base = total / size;
    int extra = total % size;
    *begin = rank * base + (rank < extra ? rank : extra);
    *end = *begin + base + (rank < extra ? 1 : 0);
}

static void step_mpi(int n, int mj, const double in[], double out[], const double w[],
                     double sgn, int rank, int size) {
    int mj2 = 2 * mj;
    int lj = n / mj2;
    int j_begin = 0;
    int j_end = 0;
    partition_range(lj, rank, size, &j_begin, &j_end);

    int local_complex = (j_end - j_begin) * mj2;
    int local_doubles = 2 * local_complex;
    double *local = NULL;
    if (local_doubles > 0) {
        local = (double *)malloc((size_t)local_doubles * sizeof(double));
        if (!local) {
            fprintf(stderr, "rank %d allocation failed in step_mpi\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    for (int j = j_begin; j < j_end; ++j) {
        int jw = j * mj;
        int ja = jw;
        int jb = n / 2 + jw;
        int jc = j * mj2;
        int local_base = (j - j_begin) * mj2;
        double wr = w[jw * 2 + 0];
        double wi = w[jw * 2 + 1];
        if (sgn < 0.0) {
            wi = -wi;
        }

        for (int k = 0; k < mj; ++k) {
            double ar = in[(ja + k) * 2 + 0];
            double ai = in[(ja + k) * 2 + 1];
            double br = in[(jb + k) * 2 + 0];
            double bi = in[(jb + k) * 2 + 1];
            double ambr = ar - br;
            double ambu = ai - bi;

            local[(local_base + k) * 2 + 0] = ar + br;
            local[(local_base + k) * 2 + 1] = ai + bi;
            local[(local_base + mj + k) * 2 + 0] = wr * ambr - wi * ambu;
            local[(local_base + mj + k) * 2 + 1] = wi * ambr + wr * ambu;
        }
    }

    int *counts = (int *)malloc((size_t)size * sizeof(int));
    int *displs = (int *)malloc((size_t)size * sizeof(int));
    if (!counts || !displs) {
        fprintf(stderr, "rank %d allocation failed for collect metadata\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int r = 0; r < size; ++r) {
        int b = 0;
        int e = 0;
        partition_range(lj, r, size, &b, &e);
        counts[r] = 2 * (e - b) * mj2;
        displs[r] = 2 * b * mj2;
    }

    MPI_Allgatherv(local, local_doubles, MPI_DOUBLE, out, counts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    free(local);
    free(counts);
    free(displs);
}
