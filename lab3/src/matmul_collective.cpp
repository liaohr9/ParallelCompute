/*
 * MPI Parallel Matrix Multiplication - Implementation 1: Collective Communication
 *
 * Approach: Row-block partitioning with MPI_Bcast + MPI_Scatter/MPI_Gather
 * - Rank 0 generates random matrices A(m×n) and B(n×k)
 * - Uses mpi_type_create_struct to pack matrix dimensions (m, n, k) into one message
 * - Broadcasts matrix B to all ranks via MPI_Bcast
 * - Scatters rows of A to all ranks via MPI_Scatter
 * - Each rank computes its partial result
 * - Gathers partial results back to rank 0 via MPI_Gather
 */

#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

// Struct to pack matrix dimensions using mpi_type_create_struct
struct MatrixDims {
    int m;
    int n;
    int k;
};

// Create MPI datatype for MatrixDims using MPI_Type_create_struct
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

// Sequential matrix multiplication: C = A * B
void matmul_seq(const double* A, const double* B, double* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int p = 0; p < n; ++p)
                sum += A[i * n + p] * B[p * k + j];
            C[i * k + j] = sum;
        }
}

// Generate random matrix with values in [-1, 1]
void random_matrix(double* M, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < rows * cols; ++i)
        M[i] = dist(gen);
}

// Print matrix (for debugging, only small matrices)
void print_matrix(const double* M, int rows, int cols, const char* name) {
    printf("=== %s (%d x %d) ===\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            printf("%10.4f ", M[i * cols + j]);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Default matrix sizes, can be overridden via command line
    int m = 512, n = 512, k = 512;
    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    // Verify dimensions are divisible by num_procs for simple block partition
    if (m % num_procs != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: m=%d is not divisible by num_procs=%d\n", m, num_procs);
        MPI_Finalize();
        return 1;
    }

    int local_m = m / num_procs; // rows per process

    // Allocate matrices on rank 0
    double *A = nullptr, *B = nullptr, *C = nullptr;
    double *local_A = nullptr, *local_C = nullptr;

    if (rank == 0) {
        A = (double*)malloc(m * n * sizeof(double));
        B = (double*)malloc(n * k * sizeof(double));
        C = (double*)malloc(m * k * sizeof(double));
        random_matrix(A, m, n);
        random_matrix(B, n, k);
    }

    local_A = (double*)malloc(local_m * n * sizeof(double));
    local_C = (double*)malloc(local_m * k * sizeof(double));

    // Create custom MPI datatype for dimensions
    MPI_Datatype dims_type = create_dims_mpi_type();

    // Pack dimensions into struct on rank 0
    MatrixDims dims;
    dims.m = m;
    dims.n = n;
    dims.k = k;

    // Broadcast dimensions using the custom struct datatype
    MPI_Bcast(&dims, 1, dims_type, 0, MPI_COMM_WORLD);

    // Broadcast the entire matrix B to all processes
    double* global_B = B;
    if (rank != 0) {
        global_B = (double*)malloc(dims.n * dims.k * sizeof(double));
    }
    MPI_Bcast(global_B, dims.n * dims.k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter rows of A to all processes
    MPI_Scatter(A, local_m * dims.n, MPI_DOUBLE,
                local_A, local_m * dims.n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Local matrix multiplication: local_C = local_A * global_B
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

    // Gather results back to rank 0
    MPI_Gather(local_C, local_m * dims.k, MPI_DOUBLE,
               C, local_m * dims.k, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // Rank 0: verify result (optional small-matrix verification) and print timing
    if (rank == 0) {
        // For small matrices, verify correctness against sequential computation
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

        // Print matrices A, B, C (only for small sizes)
        if (m <= 8 && n <= 8 && k <= 8) {
            print_matrix(A, m, n, "A");
            print_matrix(B, n, k, "B");
            print_matrix(C, m, k, "C");
        }

        free(A);
        free(B);
        free(C);
    }

    if (rank != 0) {
        free(global_B);
    }
    free(local_A);
    free(local_C);

    MPI_Type_free(&dims_type);
    MPI_Finalize();
    return 0;
}
