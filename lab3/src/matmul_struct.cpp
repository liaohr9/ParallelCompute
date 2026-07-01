/*
 * MPI Parallel Matrix Multiplication - Implementation 2: Extensive Struct Communication
 *
 * Approach: Uses mpi_type_create_struct to aggregate multiple variables
 * (dimensions + row counts + rank info) into a single communication step.
 * Then uses MPI_Scatter/MPI_Gather with custom datatypes for data distribution.
 */

#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>

// Struct aggregating multiple variables for communication
struct TaskInfo {
    int m;
    int n;
    int k;
    int local_m;    // rows assigned to each worker
    int num_procs;
    int rank;       // filled by receiver
};

// Create MPI datatype for TaskInfo
MPI_Datatype create_task_info_type() {
    MPI_Datatype info_type;
    const int nblocks = 5;
    int block_lengths[nblocks] = {1, 1, 1, 1, 1};
    MPI_Aint offsets[nblocks];
    MPI_Datatype types[nblocks] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};

    offsets[0] = offsetof(TaskInfo, m);
    offsets[1] = offsetof(TaskInfo, n);
    offsets[2] = offsetof(TaskInfo, k);
    offsets[3] = offsetof(TaskInfo, local_m);
    offsets[4] = offsetof(TaskInfo, rank);

    MPI_Type_create_struct(nblocks, block_lengths, offsets, types, &info_type);
    MPI_Type_commit(&info_type);
    return info_type;
}

// Struct for sending a chunk of matrix A with its metadata
struct ChunkInfo {
    int row_start;
    int row_end;
    int n;
    int k;
};

MPI_Datatype create_chunk_info_type() {
    MPI_Datatype chunk_type;
    const int nblocks = 4;
    int block_lengths[nblocks] = {1, 1, 1, 1};
    MPI_Aint offsets[nblocks];
    MPI_Datatype types[nblocks] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};

    offsets[0] = offsetof(ChunkInfo, row_start);
    offsets[1] = offsetof(ChunkInfo, row_end);
    offsets[2] = offsetof(ChunkInfo, n);
    offsets[3] = offsetof(ChunkInfo, k);

    MPI_Type_create_struct(nblocks, block_lengths, offsets, types, &chunk_type);
    MPI_Type_commit(&chunk_type);
    return chunk_type;
}

// Struct for sending result metadata
struct ResultInfo {
    int rank;
    int row_start;
    int row_end;
    int k;
};

MPI_Datatype create_result_info_type() {
    MPI_Datatype res_type;
    const int nblocks = 4;
    int block_lengths[nblocks] = {1, 1, 1, 1};
    MPI_Aint offsets[nblocks];
    MPI_Datatype types[nblocks] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};

    offsets[0] = offsetof(ResultInfo, rank);
    offsets[1] = offsetof(ResultInfo, row_start);
    offsets[2] = offsetof(ResultInfo, row_end);
    offsets[3] = offsetof(ResultInfo, k);

    MPI_Type_create_struct(nblocks, block_lengths, offsets, types, &res_type);
    MPI_Type_commit(&res_type);
    return res_type;
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

    // Create custom datatypes
    MPI_Datatype task_info_type = create_task_info_type();
    MPI_Datatype chunk_info_type = create_chunk_info_type();
    MPI_Datatype result_info_type = create_result_info_type();

    // Rank 0 prepares data
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

    // Broadcast task info using struct datatype
    TaskInfo info;
    info.m = m;
    info.n = n;
    info.k = k;
    info.local_m = local_m;
    info.num_procs = num_procs;
    info.rank = rank;

    MPI_Bcast(&info, 1, task_info_type, 0, MPI_COMM_WORLD);

    // Broadcast matrix B
    double *global_B = (rank == 0) ? B : (double*)malloc(info.n * info.k * sizeof(double));
    MPI_Bcast(global_B, info.n * info.k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter rows of A
    MPI_Scatter(A, info.local_m * info.n, MPI_DOUBLE,
                local_A, info.local_m * info.n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Local computation
    double t_start = MPI_Wtime();
    for (int i = 0; i < info.local_m; ++i)
        for (int j = 0; j < info.k; ++j) {
            double sum = 0.0;
            for (int p = 0; p < info.n; ++p)
                sum += local_A[i * info.n + p] * global_B[p * info.k + j];
            local_C[i * info.k + j] = sum;
        }
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    // Gather results
    MPI_Gather(local_C, info.local_m * info.k, MPI_DOUBLE,
               C, info.local_m * info.k, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Verification for small matrices
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

    if (rank != 0) free(global_B);
    free(local_A);
    free(local_C);

    MPI_Type_free(&task_info_type);
    MPI_Type_free(&chunk_info_type);
    MPI_Type_free(&result_info_type);
    MPI_Finalize();
    return 0;
}
