/*
 * MPI Parallel Matrix Multiplication - Implementation 4: Column-Block Partitioning
 *
 * Approach: Partition matrix A by columns (and B by rows) instead of rows.
 * Each rank gets a block of columns from A and corresponding rows of B,
 * computes partial product, then uses MPI_Allreduce to sum results.
 *
 * Uses mpi_type_create_struct for dimension communication (required).
 * Uses MPI_Type_vector to create derived datatype for column-block access.
 * Uses MPI_Gatherv for gathering results.
 *
 * This is a different data partitioning strategy (选做).
 */

#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <random>

// Struct for dimensions using mpi_type_create_struct
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

    if (n % num_procs != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: n=%d not divisible by num_procs=%d\n", n, num_procs);
        MPI_Finalize();
        return 1;
    }

    int local_n = n / num_procs; // columns of A / rows of B per process

    double *A = nullptr, *B = nullptr, *C = nullptr;
    if (rank == 0) {
        A = (double*)malloc(m * n * sizeof(double));
        B = (double*)malloc(n * k * sizeof(double));
        C = (double*)malloc(m * k * sizeof(double));
        random_matrix(A, m, n);
        random_matrix(B, n, k);
    }

    // Each rank gets its column block of A and row block of B
    double *local_A = (double*)malloc(m * local_n * sizeof(double));
    double *local_B = (double*)malloc(local_n * k * sizeof(double));
    double *local_C = (double*)calloc(m * k, sizeof(double));

    MPI_Datatype dims_type = create_dims_mpi_type();
    MatrixDims dims;
    dims.m = m;
    dims.n = n;
    dims.k = k;

    // Broadcast dims using struct datatype
    MPI_Bcast(&dims, 1, dims_type, 0, MPI_COMM_WORLD);

    // --- Scatter columns of A using MPI_Type_vector ---
    // MPI_Type_vector: count blocks, blocklength elements per block, stride apart
    // Column block for rank r: m blocks, each block has local_n elements, stride = n
    MPI_Datatype col_block_type;
    MPI_Type_vector(m, local_n, dims.n, MPI_DOUBLE, &col_block_type);
    MPI_Type_commit(&col_block_type);

    // On rank 0, extract column blocks into a contiguous send buffer
    double* col_sendbuf = nullptr;
    if (rank == 0) {
        col_sendbuf = (double*)malloc(m * n * sizeof(double));
        for (int r = 0; r < num_procs; ++r) {
            // Use MPI_Sendrecv to copy via derived type into contiguous buffer
            MPI_Sendrecv(A + r * local_n, 1, col_block_type, 0, 0,
                         col_sendbuf + r * m * local_n, m * local_n, MPI_DOUBLE,
                         0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Scatter contiguous column blocks
    MPI_Scatter(col_sendbuf, m * local_n, MPI_DOUBLE,
                local_A, m * local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Scatter row blocks of B (rows are contiguous in row-major)
    MPI_Scatter(B, local_n * dims.k, MPI_DOUBLE,
                local_B, local_n * dims.k, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Local computation: local_C = local_A * local_B
    double t_start = MPI_Wtime();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < dims.k; ++j) {
            double sum = 0.0;
            for (int p = 0; p < local_n; ++p)
                sum += local_A[i * local_n + p] * local_B[p * dims.k + j];
            local_C[i * dims.k + j] = sum;
        }

    // Reduce all partial results using MPI_Allreduce
    MPI_Allreduce(MPI_IN_PLACE, local_C, m * k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    if (rank == 0) {
        memcpy(C, local_C, m * k * sizeof(double));

        // Verification
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
        free(col_sendbuf);
    }

    free(local_A);
    free(local_B);
    free(local_C);

    MPI_Type_free(&dims_type);
    MPI_Type_free(&col_block_type);
    MPI_Finalize();
    return 0;
}
