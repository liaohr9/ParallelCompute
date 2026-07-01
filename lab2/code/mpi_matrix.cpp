/*
 * ============================================================
 * 基于 MPI 点对点通信的并行矩阵乘法
 * ============================================================
 *
 * 算法说明：
 *   计算 C = A × B，其中 A 为 m×n 矩阵，B 为 n×p 矩阵，C 为 m×p 矩阵。
 *
 *   采用行分块（Row-wise Block）策略：
 *   - Rank 0（主进程）负责生成随机矩阵 A(m×n) 和 B(n×p)
 *   - 将矩阵 A 的行均匀分给各进程，矩阵 B 广播给所有进程
 *   - 每个进程独立计算分配给它的 C 的行
 *   - 所有计算结果汇总回 Rank 0
 *
 * 通信模式：
 *   - Rank 0 -> 各进程（包括自己）：发送 A 的子矩阵行 + 发送完整的 B
 *   - 各进程 -> Rank 0：发送计算得到的 C 的子矩阵行
 *
 * 使用方法：
 *   编译：mpicxx -O3 -o mpi_matrix mpi_matrix.cpp
 *   运行：mpirun -np <进程数> ./mpi_matrix <m> <n> <p>
 *   示例：mpirun -np 4 ./mpi_matrix 256 256 256
 *
 * ============================================================
 */

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

using namespace std;

/* ============================================================
 * 工具函数：将 2D 矩阵展平为 1D 数组后的索引计算
 * ============================================================
 * 矩阵以行优先（Row-major）方式存储在 1D 数组中：
 *   matrix[i * cols + j] 等价于 matrix[i][j]
 */
inline int idx(int i, int j, int cols) {
    return i * cols + j;
}

/* ============================================================
 * 函数：生成随机矩阵
 * -----------------------------------------------------------
 * 参数：
 *   rows, cols - 矩阵维度
 * 返回：
 *   包含 [0, 100) 范围内随机整数的 1D 数组（行优先存储）
 * ============================================================
 */
vector<double> generate_matrix(int rows, int cols) {
    vector<double> mat(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (double)(rand() % 100);
    }
    return mat;
}

/* ============================================================
 * 函数：打印矩阵（仅用于小规模矩阵的调试/验证）
 * -----------------------------------------------------------
 * 参数：
 *   mat  - 矩阵数据（1D 数组，行优先）
 *   rows, cols - 矩阵维度
 *   label - 矩阵名称标签
 * ============================================================
 */
void print_matrix(const vector<double>& mat, int rows, int cols, const char* label) {
    printf("=== %s (%d x %d) ===\n", label, rows, cols);
    // 矩阵规模较大时仅打印前 4x4 子矩阵，避免输出过多
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
 * 函数：朴素串行矩阵乘法（用于结果验证）
 * -----------------------------------------------------------
 * 计算 C = A × B
 *   A: m×n 矩阵
 *   B: n×p 矩阵
 *   C: m×p 矩阵（输出）
 * 时间复杂度：O(m × n × p)
 * ============================================================
 */
void serial_matrix_multiply(const vector<double>& A, int m, int n,
                            const vector<double>& B, int p,
                            vector<double>& C) {
    // 初始化结果矩阵为零
    memset(C.data(), 0, sizeof(double) * m * p);

    // 三重循环实现矩阵乘法
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            double a_val = A[idx(i, k, n)];  // 提前取出 A[i][k]，减少重复索引计算
            for (int j = 0; j < p; j++) {
                C[idx(i, j, p)] += a_val * B[idx(k, j, p)];
            }
        }
    }
}

/* ============================================================
 * 函数：验证两个矩阵是否相等（允许浮点误差）
 * ============================================================
 */
bool matrices_equal(const vector<double>& C1, const vector<double>& C2, int m, int p, double tol = 1e-6) {
    for (int i = 0; i < m * p; i++) {
        if (fabs(C1[i] - C2[i]) > tol) {
            return false;
        }
    }
    return true;
}

/* ============================================================
 * 主函数
 * ============================================================
 */
int main(int argc, char** argv) {
    // ---- MPI 初始化 ----
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 当前进程编号（0 ~ size-1）
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 总进程数

    /* ============================================================
     * 1. 解析输入参数
     *    命令行参数：m n p（矩阵维度）
     *    每个参数的取值范围为 [128, 16384]
     * ============================================================
     */
    int m, n, p;
    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        p = atoi(argv[3]);
    } else {
        // 默认值：方阵 256×256×256
        m = 256;
        n = 256;
        p = 256;
    }

    // 验证参数合法性
    if (m < 128 || m > 16384 || n < 128 || n > 16384 || p < 128 || p > 16384) {
        if (rank == 0) {
            printf("错误：矩阵维度必须在 [128, 16384] 范围内，当前 m=%d, n=%d, p=%d\n", m, n, p);
        }
        MPI_Finalize();
        return 1;
    }

    /* ============================================================
     * 2. 任务分配：将矩阵 A 的 m 行划分给 size 个进程
     *
     *    使用均分策略：
     *    - base_rows = m / size   （每个进程至少分到的行数）
     *    - remainder = m % size    （前 remainder 个进程多分一行）
     *
     *    例：m=10, size=3 → 进程0分4行、进程1分3行、进程2分3行
     * ============================================================
     */
    int base_rows = m / size;        // 每个进程的基本行数
    int remainder = m % size;        // 余数，需要额外分配的行之数

    // 计算当前进程分到的行数
    int my_rows = base_rows + (rank < remainder ? 1 : 0);

    // 计算当前进程分配的起始行号（用于结果组装时定位）
    // 公式：每个进程起始行 = rank * base_rows + min(rank, remainder)
    // 因为前 remainder 个进程各多一行，所以前 rank 个进程中有 min(rank, remainder) 个多了一行
    int my_start_row = rank * base_rows + (rank < remainder ? rank : remainder);

    /* ============================================================
     * 3. Rank 0 生成随机矩阵 A(m×n) 和 B(n×p)
     * ============================================================
     */
    vector<double> A, B;
    if (rank == 0) {
        srand(42);  // 固定种子，保证每次运行结果可复现
        A = generate_matrix(m, n);
        B = generate_matrix(n, p);

        printf("================================================\n");
        printf("  MPI 并行矩阵乘法\n");
        printf("================================================\n");
        printf("  矩阵规模: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", m, n, n, p, m, p);
        printf("  MPI 进程数: %d\n", size);
        printf("  每进程分配行数: ");
        for (int r = 0; r < size; r++) {
            int rows_r = base_rows + (r < remainder ? 1 : 0);
            printf("P%d:%d ", r, rows_r);
        }
        printf("\n");
        printf("================================================\n\n");
    }

    /* ============================================================
     * 4. 通信阶段 1：Rank 0 将数据分发给各进程
     *
     *    对每个进程（包括自身），发送：
     *    a) 矩阵 A 中该进程负责的行（子矩阵）
     *    b) 完整的矩阵 B（每个进程都需要 B 的全部列来计算）
     *
     *    使用 MPI_Send / MPI_Recv 点对点通信
     * ============================================================
     */

    // ---- 每个进程接收分配到的 A 的子矩阵 ----
    vector<double> my_A(my_rows * n);  // 本地 A 子矩阵

    if (rank == 0) {
        // 进程 0 自己的数据直接从 A 中拷贝，无需通信
        memcpy(my_A.data(), A.data(), sizeof(double) * my_rows * n);

        // 向其他进程发送它们的 A 子矩阵行
        for (int dest = 1; dest < size; dest++) {
            int dest_rows = base_rows + (dest < remainder ? 1 : 0);

            // 计算该进程负责的起始行号
            int row_start = 0;
            for (int r = 0; r < dest; r++) {
                row_start += base_rows + (r < remainder ? 1 : 0);
            }

            // 发送 A 的子矩阵：从 A[row_start * n] 开始，共 dest_rows * n 个 double
            MPI_Send(&A[row_start * n], dest_rows * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        // 非 0 号进程接收自己的 A 子矩阵
        MPI_Recv(my_A.data(), my_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // ---- 每个进程接收完整的矩阵 B ----
    // B 的大小为 n×p，所有进程都需要完整的 B
    vector<double> my_B(n * p);

    if (rank == 0) {
        // Rank 0 直接从已有的 B 拷贝
        my_B = B;

        // 向其他进程发送完整的 B
        for (int dest = 1; dest < size; dest++) {
            MPI_Send(B.data(), n * p, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
        }
    } else {
        // 非 0 号进程接收 B
        MPI_Recv(my_B.data(), n * p, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* ============================================================
     * 5. 本地计算：每个进程计算自己分配的 C 的行
     *
     *    进程负责 my_rows 行，每行需要与 B(n×p) 做矩阵乘法
     *    本地计算结果存储在 my_C（大小为 my_rows × p）
     * ============================================================
     */

    // 记录计算时间（使用 MPI_Wtime 获取高精度时间戳）
    MPI_Barrier(MPI_COMM_WORLD);   // 所有进程同步，确保数据都已到位
    double start_time = MPI_Wtime();

    vector<double> my_C(my_rows * p, 0.0);  // 初始化为 0

    // 本地矩阵乘法：C[i][j] = sum(A[i][k] * B[k][j])
    for (int i = 0; i < my_rows; i++) {
        for (int k = 0; k < n; k++) {
            double a_val = my_A[idx(i, k, n)];
            for (int j = 0; j < p; j++) {
                my_C[idx(i, j, p)] += a_val * my_B[idx(k, j, p)];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);   // 同步，确保所有进程完成计算
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;  // 计算耗时（秒）

    /* ============================================================
     * 6. 通信阶段 2：各进程将计算结果发送回 Rank 0
     *
     *    每个进程发送其 my_C（大小为 my_rows × p）给 Rank 0
     *    Rank 0 将结果组装成完整的 C(m×p) 矩阵
     * ============================================================
     */
    vector<double> C;  // 完整的 C 矩阵（仅 Rank 0 需要）

    if (rank == 0) {
        C.resize(m * p);

        // 先拷贝自己的结果
        memcpy(&C[my_start_row * p], my_C.data(), sizeof(double) * my_rows * p);

        // 接收其他进程的结果并组装
        for (int src = 1; src < size; src++) {
            int src_rows = base_rows + (src < remainder ? 1 : 0);

            // 计算该进程结果在 C 中的起始行号
            int row_offset = 0;
            for (int r = 0; r < src; r++) {
                row_offset += base_rows + (r < remainder ? 1 : 0);
            }

            // 接收数据并直接写入 C 的对应位置
            MPI_Recv(&C[row_offset * p], src_rows * p, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        // 非 0 号进程发送自己的结果给 Rank 0
        MPI_Send(my_C.data(), my_rows * p, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    /* ============================================================
     * 7. Rank 0 输出结果
     * ============================================================
     */
    if (rank == 0) {
        // 打印矩阵 A, B, C（大规模时只打印前 4x4 子矩阵）
        print_matrix(A, m, n, "Matrix A");
        print_matrix(B, n, p, "Matrix B");
        print_matrix(C, m, p, "Matrix C");

        // 打印性能信息
        printf("================================================\n");
        printf("  性能统计\n");
        printf("================================================\n");
        printf("  进程数: %d\n", size);
        printf("  计算时间: %.6f 秒 (%.2f 毫秒)\n", elapsed, elapsed * 1000.0);
        printf("================================================\n");
    }

    /* ============================================================
     * 8. 结果正确性验证（可选）
     *
     *    使用串行矩阵乘法在 Rank 0 上验证结果（仅对小矩阵执行）
     * ============================================================
     */
    if (rank == 0 && m <= 512 && n <= 512 && p <= 512) {
        // 对中小规模矩阵执行串行计算以验证正确性
        vector<double> C_serial(m * p);
        serial_matrix_multiply(A, m, n, B, p, C_serial);

        if (matrices_equal(C, C_serial, m, p)) {
            printf("[验证] 结果正确 ✓\n");
        } else {
            printf("[验证] 结果错误 ✗ - 并行结果与串行结果不一致！\n");
        }
    }

    // ---- MPI 结束 ----
    MPI_Finalize();
    return 0;
}
