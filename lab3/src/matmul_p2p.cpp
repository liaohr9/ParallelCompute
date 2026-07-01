/*
 * MPI Parallel Matrix Multiplication - Implementation 3: Point-to-Point Communication
 *
 * Approach: Uses MPI_Send/MPI_Recv instead of collective operations.
 * This serves as a baseline for comparing communication overhead.
 * Also uses mpi_type_create_struct for dimension broadcast (as required).
 */

#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>

// Struct for dimensions (required by experiment)
struct MatrixDims {
    int m;
    int n;
    int k;
};

MPI_Datatype create_dims_mpi_type() {
    MPI_Datatype dims_type;
    int block_lengths[3] = {1, 1, 1};
    MPI_Aint offsets[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};

    offsets[0] = offsetof(MatrixDims, m);
    offsets[1] = offsetof(MatrixDims, n);
    offsets[2] = offsetof(MatrixDims, k);

    MPI_Type_create_struct(3, block_lengths, offsets, types, &dims_type);
    MPI_Type_commit(&dims_type);
    return dims_type;
}

void random_matrix(double* M, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < rows * cols; ++i)
        M[i] = dist(gen);
}

void matmul_seq(const double* A, const double* B, double* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int p = 0; p < n; ++p)
                sum += A[i * n + p] * B[p * k + j];
            C[i * k + j] = sum;
        }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int m = 512, n = 512, k = 512;
    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    if (m % num_procs != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: m=%d not divisible by num_procs=%d\n", m, num_procs);
        MPI_Finalize();
        return 1;
    }

    int local_m = m / num_procs;

    // Rank 0 allocates and initializes matrices
    double *A = nullptr, *B = nullptr, *C = nullptr;
    if (rank == 0) {
        A = (double*)malloc(m * n * sizeof(double));
        B = (double*)malloc(n * k * sizeof(double));
        C = (double*)malloc(m * k * sizeof(double));
        random_matrix(A, m, n);
        random_matrix(B, n, k);
    }

    double *local_A = (double*)malloc(local_m * n * sizeof(double));
    double *local_C = (double*)malloc(local_m * k * sizeof(double));
    double *global_B = (double*)malloc(n * k * sizeof(double));

    MPI_Datatype dims_type = create_dims_mpi_type();
    MatrixDims dims;
    dims.m = m;
    dims.n = n;
    dims.k = k;

    // Broadcast dims using custom struct datatype
    MPI_Bcast(&dims, 1, dims_type, 0, MPI_COMM_WORLD);

    // --- Point-to-Point Communication ---

    // Rank 0 sends matrix B to all workers
    if (rank == 0) {
        for (int r = 1; r < num_procs; ++r)
            MPI_Send(B, dims.n * dims.k, MPI_DOUBLE, r, 100, MPI_COMM_WORLD);
        memcpy(global_B, B, dims.n * dims.k * sizeof(double));
    } else {
        MPI_Recv(global_B, dims.n * dims.k, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Rank 0 sends chunks of A to each worker
    if (rank == 0) {
        memcpy(local_A, A, local_m * dims.n * sizeof(double));
        for (int r = 1; r < num_procs; ++r)
            MPI_Send(A + r * local_m * dims.n, local_m * dims.n, MPI_DOUBLE, r, 200, MPI_COMM_WORLD);
    } else {
        MPI_Recv(local_A, local_m * dims.n, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Local computation
    double t_start = MPI_Wtime();
    for (int i = 0; i < local_m; ++i)
        for (int j = 0; j < dims.k; ++j) {
            double sum = 0.0;
            for (int p = 0; p < dims.n; ++p)
                sum += local_A[i * dims.n + p] * global_B[p * dims.k + j];
            local_C[i * dims.k + j] = sum;
        }
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    // Rank 0 receives results from each worker
    if (rank == 0) {
        memcpy(C, local_C, local_m * dims.k * sizeof(double));
        for (int r = 1; r < num_procs; ++r)
            MPI_Recv(C + r * local_m * dims.k, local_m * dims.k, MPI_DOUBLE, r, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Send(local_C, local_m * dims.k, MPI_DOUBLE, 0, 300, MPI_COMM_WORLD);
    }

    // Verification
    if (rank == 0) {
        if (m <= 256 && n <= 256 && k <= 256) {
            double* C_ref = (double*)malloc(m * k * sizeof(double));
            matmul_seq(A, B, C_ref, m, n, k);
            double max_err = 0.0;
            for (int i = 0; i < m * k; ++i) {
                double err = fabs(C[i] - C_ref[i]);
                if (err > max_err) max_err = err;
            }
            printf("Verification: max_error = %.2e\n", max_err);
            free(C_ref);
        }

        printf("Matrix multiplication: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", m, n, n, k, m, k);
        printf("Processes: %d\n", num_procs);
        printf("Time (compute only): %.6f seconds\n", elapsed);

        free(A);
        free(B);
        free(C);
    }

    free(global_B);
    free(local_A);
    free(local_C);

    MPI_Type_free(&dims_type);
    MPI_Finalize();
    return 0;
}
