/*
 * ============================================================
 * 基于 CSR 格式的稀疏矩阵并行乘法
 * ============================================================
 *
 * 算法说明：
 *   计算 C = A × B，其中 A 为稀疏矩阵（CSR 格式），B 为稠密矩阵。
 *
 *   CSR (Compressed Sparse Row) 存储格式：
 *   - values[]      : 非零元素的值（按行优先）
 *   - col_indices[] : 每个非零元素对应的列索引
 *   - row_ptr[]     : 每行的非零元素在 values 中的起始位置
 *     （row_ptr[i+1] - row_ptr[i] = 第 i 行的非零元数）
 *
 *   稀疏矩阵生成：使用稀疏度（sparsity）参数控制非零元素比例。
 *
 *   并行策略：
 *   - Rank 0 生成稀疏矩阵 A（CSR）和稠密矩阵 B
 *   - A 按行分发给各进程（CSR 格式）
 *   - B 广播给所有进程
 *   - 各进程计算本地行：只遍历非零元素
 *
 * 性能优势：
 *   密集方案：O(m × n × p) 计算，O(m × n) 存储
 *   稀疏方案：O(nnz(A) × p) 计算，O(nnz(A)) 存储
 *   当稀疏度 = 1% 时，计算量减少 ~100 倍
 *
 * 使用方法：
 *   编译：mpicxx -O3 -o mpi_sparse mpi_sparse.cpp
 *   运行：mpirun -np 4 ./mpi_sparse 2048 0.01
 *   参数：
 *     m    : 矩阵维度（方阵，A/B/C 都是 m×m）
 *     sparsity : 稀疏度（0~1，表示非零元素比例）
 */

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>

using namespace std;

/* ============================================================
 * CSR 矩阵结构
 * ============================================================
 */
struct CSRMatrix {
    vector<double> values;       // 非零元素值
    vector<int>    col_indices;  // 列索引
    vector<int>    row_ptr;      // 行起始指针（大小 = rows + 1）
    int rows;
    int cols;
    int nnz;                     // 非零元素总数
};

/* ============================================================
 * 生成稀疏随机矩阵（CSR 格式）
 * ============================================================
 */
CSRMatrix generate_sparse_matrix(int rows, int cols, double sparsity) {
    CSRMatrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.nnz = 0;

    mat.row_ptr.reserve(rows + 1);
    mat.row_ptr.push_back(0);

    for (int i = 0; i < rows; i++) {
        /* 每行非零元素：按二项分布近似 */
        int count = 0;
        for (int j = 0; j < cols; j++) {
            if ((double)rand() / RAND_MAX < sparsity) {
                mat.values.push_back((double)(rand() % 100));
                mat.col_indices.push_back(j);
                count++;
            }
        }
        mat.row_ptr.push_back(mat.row_ptr.back() + count);
    }
    mat.nnz = (int)mat.values.size();
    return mat;
}

/* ============================================================
 * 生成稠密随机矩阵
 * ============================================================
 */
vector<double> generate_dense_matrix(int rows, int cols) {
    vector<double> mat(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (double)(rand() % 100);
    }
    return mat;
}

inline int idx(int i, int j, int cols) {
    return i * cols + j;
}

void print_matrix(const vector<double>& mat, int rows, int cols, const char* label) {
    printf("=== %s (%d x %d) ===\n", label, rows, cols);
    int pr = rows < 4 ? rows : 4;
    int pc = cols < 4 ? cols : 4;
    if (rows > 4 || cols > 4) {
        printf("（仅显示前 %dx%d 子矩阵）\n", pr, pc);
    }
    for (int i = 0; i < pr; i++) {
        for (int j = 0; j < pc; j++) {
            printf("%8.1f ", mat[idx(i, j, cols)]);
        }
        printf("\n");
    }
    printf("\n");
}

/* ============================================================
 * 稠密矩阵乘法（参考实现，用于验证）
 * ============================================================
 */
void serial_dense_multiply(const vector<double>& A, int m, int n,
                           const vector<double>& B, int p,
                           vector<double>& C) {
    memset(C.data(), 0, sizeof(double) * m * p);
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            double a_val = A[idx(i, k, n)];
            for (int j = 0; j < p; j++) {
                C[idx(i, j, p)] += a_val * B[idx(k, j, p)];
            }
        }
    }
}

/* ============================================================
 * CSR × 稠密 矩阵乘法
 * C = A(CSR) × B(dense)，结果 C 为稠密矩阵
 * 复杂度：O(nnz(A) × p)，而非 O(m × n × p)
 * ============================================================
 */
void csr_dense_multiply(const CSRMatrix& A, const vector<double>& B,
                        int p, vector<double>& C, int n) {
    memset(C.data(), 0, sizeof(double) * A.rows * p);

    for (int i = 0; i < A.rows; i++) {
        int row_start = A.row_ptr[i];
        int row_end   = A.row_ptr[i + 1];
        for (int k = row_start; k < row_end; k++) {
            double val = A.values[k];
            int col = A.col_indices[k];
            for (int j = 0; j < p; j++) {
                C[idx(i, j, p)] += val * B[idx(col, j, p)];
            }
        }
    }
}

/* ============================================================
 * 将稠密矩阵转为 CSR 格式（用于生成参考结果）
 * ============================================================
 */
CSRMatrix dense_to_csr(const vector<double>& A, int m, int n) {
    CSRMatrix mat;
    mat.rows = m;
    mat.cols = n;
    mat.row_ptr.reserve(m + 1);
    mat.row_ptr.push_back(0);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double val = A[idx(i, j, n)];
            if (val != 0.0) {
                mat.values.push_back(val);
                mat.col_indices.push_back(j);
            }
        }
        mat.row_ptr.push_back((int)mat.values.size());
    }
    mat.nnz = (int)mat.values.size();
    return mat;
}

bool matrices_equal(const vector<double>& C1, const vector<double>& C2, int m, int p, double tol = 1e-6) {
    for (int i = 0; i < m * p; i++) {
        if (fabs(C1[i] - C2[i]) > tol) {
            return false;
        }
    }
    return true;
}

/* ============================================================
 * 序列化 CSR 矩阵为 flat 数组（用于 MPI 通信）
 * ============================================================
 */
struct CSRPack {
    int rows;
    int cols;
    int nnz;
    vector<double> values;
    vector<int>    col_indices;
    vector<int>    row_ptr;  // size = rows + 1
};

CSRPack pack_csr(const CSRMatrix& mat) {
    CSRPack p;
    p.rows = mat.rows;
    p.cols = mat.cols;
    p.nnz  = mat.nnz;
    p.values = mat.values;
    p.col_indices = mat.col_indices;
    p.row_ptr = mat.row_ptr;
    return p;
}

/* ============================================================
 * 从 CSR 中按行范围提取子矩阵
 * ============================================================
 */
CSRMatrix extract_rows(const CSRMatrix& mat, int start_row, int num_rows) {
    CSRMatrix sub;
    sub.rows = num_rows;
    sub.cols = mat.cols;
    sub.row_ptr.reserve(num_rows + 1);
    sub.row_ptr.push_back(0);

    for (int i = 0; i < num_rows; i++) {
        int src_row = start_row + i;
        int src_start = mat.row_ptr[src_row];
        int src_end   = mat.row_ptr[src_row + 1];
        for (int k = src_start; k < src_end; k++) {
            sub.values.push_back(mat.values[k]);
            sub.col_indices.push_back(mat.col_indices[k]);
        }
        sub.row_ptr.push_back((int)sub.values.size());
    }
    sub.nnz = (int)sub.values.size();
    return sub;
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* 解析参数 */
    if (argc < 3) {
        if (rank == 0) {
            printf("用法: mpirun -np <N> %s <m> <sparsity>\n", argv[0]);
            printf("  m          : 矩阵维度（方阵）\n");
            printf("  sparsity   : 非零元素比例 (0~1)，例 0.01 = 1%% 非零\n");
            printf("\n示例:\n");
            printf("  mpirun -np 4 %s 2048 0.01   # 1%% 稀疏度\n", argv[0]);
            printf("  mpirun -np 4 %s 4096 0.001  # 0.1%% 稀疏度\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int m = atoi(argv[1]);
    double sparsity = atof(argv[2]);

    if (m < 128 || m > 16384) {
        if (rank == 0) {
            printf("错误：矩阵维度必须在 [128, 16384] 范围内，当前 m=%d\n", m);
        }
        MPI_Finalize();
        return 1;
    }
    if (sparsity <= 0.0 || sparsity > 1.0) {
        if (rank == 0) {
            printf("错误：稀疏度必须在 (0, 1] 范围内，当前 sparsity=%.4f\n", sparsity);
        }
        MPI_Finalize();
        return 1;
    }

    /* ---- Rank 0 生成矩阵 ---- */
    CSRMatrix A;
    vector<double> B;

    if (rank == 0) {
        srand(42);
        A = generate_sparse_matrix(m, m, sparsity);
        B = generate_dense_matrix(m, m);

        printf("================================================\n");
        printf("  稀疏矩阵并行乘法 (CSR)\n");
        printf("================================================\n");
        printf("  矩阵规模: A(%d x %d) [稀疏] * B(%d x %d) [稠密] = C(%d x %d)\n", m, m, m, m, m, m);
        printf("  稀疏度: %.2f%% (%.1f%% 非零)\n", sparsity * 100, sparsity * 100);
        printf("  A 的非零元素数 (nnz): %d\n", A.nnz);
        printf("  A 内存(CSR): %.2f MB\n",
               (A.nnz * sizeof(double) + A.nnz * sizeof(int) + (m + 1) * sizeof(int)) / (1024.0 * 1024.0));
        printf("  MPI 进程数: %d\n", size);
        printf("================================================\n\n");
    }

    /* ---- 分发矩阵 A（CSR） ---- */
    /* 计算每个进程分到的行数 */
    int base_rows = m / size;
    int remainder = m % size;
    int my_rows = base_rows + (rank < remainder ? 1 : 0);
    /* 当前进程负责的起始行 */

    CSRMatrix my_A;
    vector<double> my_B(m * m);

    if (rank == 0) {
        /* 进程 0 自己的 CSR 子矩阵 */
        my_A = extract_rows(A, 0, my_rows);

        /* 发送 CSR 子矩阵给其他进程 */
        for (int dest = 1; dest < size; dest++) {
            int dest_rows = base_rows + (dest < remainder ? 1 : 0);
            int dest_start = dest * base_rows + (dest < remainder ? dest : remainder);

            CSRMatrix sub = extract_rows(A, dest_start, dest_rows);
            CSRPack packed = pack_csr(sub);

            /* 先发送元数据 */
            int meta[3] = {packed.rows, packed.cols, packed.nnz};
            MPI_Send(meta, 3, MPI_INT, dest, 0, MPI_COMM_WORLD);

            /* 发送 values */
            if (packed.nnz > 0) {
                MPI_Send(packed.values.data(), packed.nnz, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
                MPI_Send(packed.col_indices.data(), packed.nnz, MPI_INT, dest, 2, MPI_COMM_WORLD);
            }

            /* 发送 row_ptr */
            MPI_Send(packed.row_ptr.data(), packed.rows + 1, MPI_INT, dest, 3, MPI_COMM_WORLD);
        }

        /* 释放完整 A */
        { CSRMatrix tmp; tmp.rows=0; tmp.cols=0; tmp.nnz=0; A = tmp; }
    } else {
        /* 接收元数据 */
        int meta[3];
        MPI_Recv(meta, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        my_A.rows = meta[0];
        my_A.cols = meta[1];
        my_A.nnz  = meta[2];
        my_A.row_ptr.reserve(my_A.rows + 1);

        if (my_A.nnz > 0) {
            my_A.values.resize(my_A.nnz);
            my_A.col_indices.resize(my_A.nnz);
            MPI_Recv(my_A.values.data(), my_A.nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(my_A.col_indices.data(), my_A.nnz, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        my_A.row_ptr.resize(my_A.rows + 1);
        MPI_Recv(my_A.row_ptr.data(), my_A.rows + 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* ---- 广播矩阵 B ---- */
    if (rank == 0) {
        memcpy(my_B.data(), B.data(), sizeof(double) * m * m);
        /* 释放完整 B */
        vector<double>().swap(B);
    }
    MPI_Bcast(my_B.data(), m * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ---- 打印前 4x4（小规模时） ---- */
    if (m <= 64 && rank == 0) {
        /* 打印 A 和 B 的前 4x4 */
        print_matrix(my_B, m, m, "Matrix B");
    }

    /* ---- 本地稀疏乘法 ---- */
    vector<double> my_C(my_A.rows * m);

    double t_start = MPI_Wtime();
    csr_dense_multiply(my_A, my_B, m, my_C, m);
    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ---- 收集结果 ---- */
    if (rank == 0) {
        vector<double> C(m * m);
        /* 拷贝自己的结果 */
        memcpy(&C[0], my_C.data(), sizeof(double) * my_rows * m);

        /* 接收其他进程的结果 */
        for (int src = 1; src < size; src++) {
            int src_rows = base_rows + (src < remainder ? 1 : 0);
            int src_start = src * base_rows + (src < remainder ? src : remainder);
            MPI_Recv(&C[src_start * m], src_rows * m, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        printf("================================================\n");
        printf("  性能统计\n");
        printf("================================================\n");
        printf("  进程数: %d\n", size);
        printf("  计算时间: %.6f 秒 (%.2f 毫秒)\n", max_time, max_time * 1000);
        printf("================================================\n");

        /* ---- 打印结果 ---- */
        print_matrix(C, m, m, "Result C");

        /* ---- 验证 ---- */
        printf("[验证] 正在生成参考结果... ");
        srand(42);
        CSRMatrix ref_A = generate_sparse_matrix(m, m, sparsity);
        vector<double> ref_B = generate_dense_matrix(m, m);
        vector<double> ref_C(m * m, 0);

        double t_ref_start = MPI_Wtime();
        csr_dense_multiply(ref_A, ref_B, m, ref_C, m);
        double t_ref_end = MPI_Wtime();
        printf("参考计算耗时 %.2f 秒\n", t_ref_end - t_ref_start);

        if (matrices_equal(C, ref_C, m, m)) {
            printf("[验证] 结果正确 ✓\n");
        } else {
            printf("[验证] 结果错误 ✗\n");
            print_matrix(C, m, m, "Parallel C");
            print_matrix(ref_C, m, m, "Serial C");
        }

        /* 同时和稠密参考对比（证明稀疏乘法等价） */
        printf("\n[对比] 正在计算稠密参考结果... ");
        vector<double> A_dense(m * m, 0);
        for (int i = 0; i < ref_A.rows; i++) {
            for (int k = ref_A.row_ptr[i]; k < ref_A.row_ptr[i + 1]; k++) {
                A_dense[idx(i, ref_A.col_indices[k], m)] = ref_A.values[k];
            }
        }
        vector<double> ref_C2(m * m);
        serial_dense_multiply(A_dense, m, m, ref_B, m, ref_C2);

        if (matrices_equal(C, ref_C2, m, m)) {
            printf("与稠密参考一致 ✓\n");
        } else {
            printf("与稠密参考不一致 ✗\n");
        }

        /* ---- 加速比分析 ---- */
        double dense_flops = (double)m * m * m * 2;  // 每次乘加 = 2 flops
        double sparse_flops = (double)ref_A.nnz * m * 2;
        double theoretical_speedup = dense_flops / sparse_flops;

        printf("\n================================================\n");
        printf("  稀疏 vs 稠密对比\n");
        printf("================================================\n");
        printf("  稠密计算量: %.2e FLOPs (O(m³))\n", dense_flops);
        printf("  稀疏计算量: %.2e FLOPs (O(nnz×m))\n", sparse_flops);
        printf("  理论加速比: %.1fx\n", theoretical_speedup);
        printf("================================================\n");
    } else {
        MPI_Send(my_C.data(), my_rows * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
