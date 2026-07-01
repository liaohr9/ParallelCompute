/*
 * ============================================================
 * Cannon 算法并行矩阵乘法
 * ============================================================
 *
 * 算法说明：
 *   计算 C = A × B，其中 A 为 m×m 方阵，B 为 m×m 方阵。
 *
 *   Cannon 算法是一种二维分块（2D Block）算法：
 *   - 进程组织成 sqrt(P) × sqrt(P) 的网格
 *   - 每个进程只持有 (m/q) × (m/q) 的子矩阵块（q = sqrt(P)）
 *   - A 和 B 都只存储 1/P 的内存（对比原始方案需要完整 B）
 *   - 通过循环移位（shift）完成计算
 *
 * 内存优势：
 *   原始方案：每进程 O(m²) 内存（B 矩阵完整拷贝）
 *   Cannon 方案：每进程 O(m²/P) 内存（A 和 B 都只存本地块）
 *
 * 通信模式：
 *   1. 初始偏移：A 向左循环移位 rank 列，B 向上循环移位 rank 行
 *   2. 循环 q 次：
 *      a) 本地矩阵乘法
 *      b) A 向左循环移位 1 步
 *      c) B 向上循环移位 1 步
 *
 * 使用方法：
 *   编译：mpicxx -O3 -o mpi_cannon mpi_cannon.cpp
 *   运行：mpirun -np 4 ./mpi_cannon 1024
 *
 *   注意：进程数必须是完全平方数（1, 4, 9, 16, 25, 36, ...）
 *   矩阵必须是方阵（Cannon 算法的要求）
 */

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

using namespace std;

inline int idx(int i, int j, int cols) {
    return i * cols + j;
}

vector<double> generate_matrix(int rows, int cols) {
    vector<double> mat(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (double)(rand() % 100);
    }
    return mat;
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

void serial_matrix_multiply(const vector<double>& A, int m, int n,
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

bool matrices_equal(const vector<double>& C1, const vector<double>& C2, int m, int p, double tol = 1e-6) {
    for (int i = 0; i < m * p; i++) {
        if (fabs(C1[i] - C2[i]) > tol) {
            return false;
        }
    }
    return true;
}

/* ============================================================
 * 循环移位：向左发送 A 子矩阵
 * ============================================================
 */
void shift_left(int rank, int q, int block, vector<double>& A_block) {
    int left  = (rank / q) * q + (rank % q + q - 1) % q;  // 同行左边（循环）
    int right = (rank / q) * q + (rank % q + 1) % q;      // 同行右边（循环）

    MPI_Sendrecv_replace(A_block.data(), block * block, MPI_DOUBLE,
                         left, 0, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/* ============================================================
 * 循环移位：向上发送 B 子矩阵
 * ============================================================
 */
void shift_up(int rank, int q, int block, vector<double>& B_block) {
    int up   = ((rank / q + q - 1) % q) * q + (rank % q);  // 同列上边（循环）
    int down = ((rank / q + 1) % q) * q + (rank % q);      // 同列下边（循环）

    MPI_Sendrecv_replace(B_block.data(), block * block, MPI_DOUBLE,
                         up, 0, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* 检查进程数是否为完全平方数 */
    int q = 0;
    for (int i = 1; i * i <= size; i++) {
        if (i * i == size) { q = i; break; }
    }
    if (q == 0) {
        if (rank == 0) {
            printf("错误：Cannon 算法要求进程数为完全平方数（当前 size=%d）\n", size);
            printf("支持：1, 4, 9, 16, 25, 36, ...\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* 解析参数 */
    if (argc < 2) {
        if (rank == 0) {
            printf("用法: mpirun -np <P^2> %s <m> [n] [p]\n", argv[0]);
            printf("  默认 n=m, p=m（方阵）\n");
        }
        MPI_Finalize();
        return 1;
    }

    int m = atoi(argv[1]);
    int n = (argc >= 3) ? atoi(argv[2]) : m;
    int p = (argc >= 4) ? atoi(argv[3]) : m;

    /* Cannon 算法要求方阵 */
    if (m != n || n != p || m % q != 0) {
        if (rank == 0) {
            printf("错误：Cannon 算法要求方阵且 m %% q == 0\n");
            printf("  当前 m=%d, n=%d, p=%d, q=%d\n", m, n, p, q);
        }
        MPI_Finalize();
        return 1;
    }

    if (m < 128 || m > 16384) {
        if (rank == 0) {
            printf("错误：矩阵维度必须在 [128, 16384] 范围内，当前 m=%d\n", m);
        }
        MPI_Finalize();
        return 1;
    }

    int block = m / q;  // 每个进程的子矩阵块大小

    /* ---- Rank 0 生成完整矩阵 ---- */
    vector<double> A, B;
    if (rank == 0) {
        srand(42);
        A = generate_matrix(m, m);
        B = generate_matrix(m, m);
    }

    /* ---- 打印信息 ---- */
    if (rank == 0) {
        printf("================================================\n");
        printf("  Cannon 算法并行矩阵乘法\n");
        printf("================================================\n");
        printf("  矩阵规模: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", m, n, n, p, m, p);
        printf("  进程网格: %d x %d (%d 进程)\n", q, q, size);
        printf("  子矩阵块: %d x %d\n", block, block);
        printf("  每进程内存: ~%.2f MB (原始方案需要 %.2f MB)\n",
               3.0 * block * block * sizeof(double) / (1024 * 1024),
               3.0 * m * m * sizeof(double) / (1024 * 1024));
        printf("================================================\n\n");
    }

    /* ---- 分发：Rank 0 将 A 和 B 的子块发送给各进程 ---- */
    vector<double> A_block(block * block);
    vector<double> B_block(block * block);

    if (rank == 0) {
        /* 进程 0 在网格中的位置 (0, 0) */
        for (int i = 0; i < block; i++) {
            for (int j = 0; j < block; j++) {
                A_block[idx(i, j, block)] = A[idx(i, j, m)];
                B_block[idx(i, j, block)] = B[idx(i, j, m)];
            }
        }

        /* 发送给其他进程 */
        for (int r = 0; r < q; r++) {
            for (int c = 0; c < q; c++) {
                if (r == 0 && c == 0) continue;  /* 跳过自己 */
                int dest = r * q + c;
                int row_start = r * block;
                int col_start = c * block;

                vector<double> send_A(block * block);
                for (int i = 0; i < block; i++) {
                    for (int j = 0; j < block; j++) {
                        send_A[idx(i, j, block)] = A[idx(row_start + i, col_start + j, m)];
                    }
                }
                MPI_Send(send_A.data(), block * block, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);

                vector<double> send_B(block * block);
                for (int i = 0; i < block; i++) {
                    for (int j = 0; j < block; j++) {
                        send_B[idx(i, j, block)] = B[idx(row_start + i, col_start + j, m)];
                    }
                }
                MPI_Send(send_B.data(), block * block, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            }
        }

        /* Rank 0 释放完整矩阵，节省内存 */
        vector<double>().swap(A);
        vector<double>().swap(B);
    } else {
        MPI_Recv(A_block.data(), block * block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_block.data(), block * block, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* ---- 初始偏移（Initial Alignment） ---- */
    /* A 向左循环移位 (行坐标) 步 */
    int my_row = rank / q;
    for (int s = 0; s < my_row; s++) {
        shift_left(rank, q, block, A_block);
    }
    /* B 向上循环移位 (列坐标) 步 */
    int my_col = rank % q;
    for (int s = 0; s < my_col; s++) {
        shift_up(rank, q, block, B_block);
    }

    /* ---- Cannon 主循环 ---- */
    vector<double> C_block(block * block, 0.0);

    double t_start = MPI_Wtime();

    for (int k = 0; k < q; k++) {
        /* 本地矩阵乘法：C_block += A_block × B_block */
        for (int i = 0; i < block; i++) {
            for (int l = 0; l < block; l++) {
                double a_val = A_block[idx(i, l, block)];
                for (int j = 0; j < block; j++) {
                    C_block[idx(i, j, block)] += a_val * B_block[idx(l, j, block)];
                }
            }
        }

        /* 移位 */
        shift_left(rank, q, block, A_block);
        shift_up(rank, q, block, B_block);
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ---- 收集结果到 Rank 0 ---- */
    if (rank == 0) {
        vector<double> C(m * m);
        /* 拷贝自己的块 */
        for (int i = 0; i < block; i++) {
            for (int j = 0; j < block; j++) {
                C[idx(i, j, m)] = C_block[idx(i, j, block)];
            }
        }

        /* 接收其他进程的块 */
        for (int r = 0; r < q; r++) {
            for (int c = 0; c < q; c++) {
                if (r == 0 && c == 0) continue;
                int src = r * q + c;
                vector<double> recv_block(block * block);
                MPI_Recv(recv_block.data(), block * block, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int row_start = r * block;
                int col_start = c * block;
                for (int i = 0; i < block; i++) {
                    for (int j = 0; j < block; j++) {
                        C[idx(row_start + i, col_start + j, m)] = recv_block[idx(i, j, block)];
                    }
                }
            }
        }

        printf("================================================\n");
        printf("  性能统计\n");
        printf("================================================\n");
        printf("  进程数: %d\n", size);
        printf("  计算时间: %.6f 秒 (%.2f 毫秒)\n", max_time, max_time * 1000);
        printf("================================================\n");

        /* ---- 验证：与串行结果对比 ---- */
        /* 重新生成矩阵用于验证 */
        srand(42);
        vector<double> ref_A = generate_matrix(m, m);
        vector<double> ref_B = generate_matrix(m, m);
        vector<double> ref_C(m * m);

        printf("[验证] 正在计算参考结果... ");
        double t_ref = MPI_Wtime();
        serial_matrix_multiply(ref_A, m, m, ref_B, m, ref_C);
        double t_ref_end = MPI_Wtime();
        printf("参考计算耗时 %.2f 秒\n", t_ref_end - t_ref);

        if (matrices_equal(C, ref_C, m, m)) {
            printf("[验证] 结果正确 ✓\n");
        } else {
            printf("[验证] 结果错误 ✗\n");
            /* 打印前 4x4 对比 */
            print_matrix(C, m, m, "Cannon C");
            print_matrix(ref_C, m, m, "Serial C");
        }
    } else {
        /* 发送结果块给 Rank 0 */
        MPI_Send(C_block.data(), block * block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
