#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int kSeedA = 1234;
constexpr int kSeedB = 1235;

enum class Kernel {
    Global2D,
    Global1D,
    Global2DBTrans,
    SharedTiled,
};

struct Config {
    int m = 512;
    int n = 512;
    int k = 512;
    int block_x = 16;
    int block_y = 16;
    int tile_k = 0;
    int warmup = 4;
    int repeats = 12;
    int verify_samples = 64;
    Kernel kernel = Kernel::SharedTiled;
    std::string kernel_name = "shared_tiled";
};

struct Timing {
    double avg_ms = 0.0;
    float min_ms = 0.0f;
    float max_ms = 0.0f;
};

struct Verification {
    bool ok = true;
    double max_abs_error = 0.0;
};

// 2D task split: one CUDA thread computes one C[row, col].
// Threads adjacent in x write adjacent C columns, which is good for coalesced writes.
__global__ void matmul_global2d(const float *__restrict__ a,
                                const float *__restrict__ b,
                                float *__restrict__ c,
                                int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= k) {
        return;
    }

    float sum = 0.0f;
    for (int p = 0; p < n; ++p) {
        sum += a[row * n + p] * b[p * k + col];
    }
    c[row * k + col] = sum;
}

// 1D task split: flatten C into a single array.
// This is included to compare task partitioning against the 2D mapping.
__global__ void matmul_global1d(const float *__restrict__ a,
                                const float *__restrict__ b,
                                float *__restrict__ c,
                                int m, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * k) {
        return;
    }

    int row = idx / k;
    int col = idx % k;
    float sum = 0.0f;
    for (int p = 0; p < n; ++p) {
        sum += a[row * n + p] * b[p * k + col];
    }
    c[idx] = sum;
}

// Same 2D task split, but B is stored as B^T on the device.
// This isolates the effect of changing memory layout from changing task layout.
__global__ void matmul_global2d_btrans(const float *__restrict__ a,
                                       const float *__restrict__ bt,
                                       float *__restrict__ c,
                                       int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= k) {
        return;
    }

    const float *a_row = a + row * n;
    const float *bt_row = bt + col * n;
    float sum = 0.0f;
    for (int p = 0; p < n; ++p) {
        sum += a_row[p] * bt_row[p];
    }
    c[row * k + col] = sum;
}

// Shared-memory tiled version.
// Each block loads one A tile and one B tile, then all threads reuse them.
__global__ void matmul_shared_tiled(const float *__restrict__ a,
                                    const float *__restrict__ b,
                                    float *__restrict__ c,
                                    int m, int n, int k,
                                    int tile_k) {
    extern __shared__ float shared[];
    float *a_tile = shared;
    float *b_tile = shared + blockDim.y * tile_k;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    float sum = 0.0f;

    for (int tile = 0; tile < n; tile += tile_k) {
        for (int idx = tid; idx < blockDim.y * tile_k; idx += threads) {
            int local_row = idx / tile_k;
            int local_col = idx % tile_k;
            int global_row = blockIdx.y * blockDim.y + local_row;
            int global_col = tile + local_col;
            a_tile[idx] = (global_row < m && global_col < n)
                              ? a[global_row * n + global_col]
                              : 0.0f;
        }

        for (int idx = tid; idx < tile_k * blockDim.x; idx += threads) {
            int local_row = idx / blockDim.x;
            int local_col = idx % blockDim.x;
            int global_row = tile + local_row;
            int global_col = blockIdx.x * blockDim.x + local_col;
            b_tile[idx] = (global_row < n && global_col < k)
                              ? b[global_row * k + global_col]
                              : 0.0f;
        }

        __syncthreads();
        for (int p = 0; p < tile_k; ++p) {
            sum += a_tile[threadIdx.y * tile_k + p] *
                   b_tile[p * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < k) {
        c[row * k + col] = sum;
    }
}

static void print_usage(const char *program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --m N --n N --k N             Matrix sizes in [128, 2048]\n"
        << "  --kernel NAME                 global2d | global1d | global2d_btrans | shared_tiled\n"
        << "  --block-x N --block-y N        CUDA block shape\n"
        << "  --tile-k N                     shared_tiled reduction tile, default block-x\n"
        << "  --warmup N --repeats N         timing iterations\n"
        << "  --verify-samples N             sampled CPU checks\n";
}

static std::string read_value(int &i, int argc, char **argv) {
    if (i + 1 >= argc) {
        std::cerr << "Missing value after " << argv[i] << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return argv[++i];
}

static int read_int(int &i, int argc, char **argv) {
    return std::stoi(read_value(i, argc, argv));
}

static Kernel parse_kernel(const std::string &name) {
    if (name == "global2d") {
        return Kernel::Global2D;
    }
    if (name == "global1d") {
        return Kernel::Global1D;
    }
    if (name == "global2d_btrans") {
        return Kernel::Global2DBTrans;
    }
    if (name == "shared_tiled") {
        return Kernel::SharedTiled;
    }
    std::cerr << "Unknown kernel: " << name << std::endl;
    std::exit(EXIT_FAILURE);
}

static Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--m") {
            cfg.m = read_int(i, argc, argv);
        } else if (arg == "--n") {
            cfg.n = read_int(i, argc, argv);
        } else if (arg == "--k") {
            cfg.k = read_int(i, argc, argv);
        } else if (arg == "--block-x") {
            cfg.block_x = read_int(i, argc, argv);
        } else if (arg == "--block-y") {
            cfg.block_y = read_int(i, argc, argv);
        } else if (arg == "--tile-k") {
            cfg.tile_k = read_int(i, argc, argv);
        } else if (arg == "--warmup") {
            cfg.warmup = read_int(i, argc, argv);
        } else if (arg == "--repeats") {
            cfg.repeats = read_int(i, argc, argv);
        } else if (arg == "--verify-samples") {
            cfg.verify_samples = read_int(i, argc, argv);
        } else if (arg == "--kernel") {
            cfg.kernel_name = read_value(i, argc, argv);
            cfg.kernel = parse_kernel(cfg.kernel_name);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    if (cfg.tile_k == 0) {
        cfg.tile_k = cfg.block_x;
    }
    return cfg;
}

static void validate_config(const Config &cfg) {
    auto require_size = [](int value, const char *name) {
        if (value < 128 || value > 2048) {
            std::cerr << name << " must be in [128, 2048], got " << value << std::endl;
            std::exit(EXIT_FAILURE);
        }
    };
    require_size(cfg.m, "m");
    require_size(cfg.n, "n");
    require_size(cfg.k, "k");

    if (cfg.block_x <= 0 || cfg.block_y <= 0 || cfg.block_x * cfg.block_y > 1024) {
        std::cerr << "block_x and block_y must be positive, and block_x * block_y <= 1024\n";
        std::exit(EXIT_FAILURE);
    }
    if (cfg.tile_k <= 0 || (cfg.kernel == Kernel::SharedTiled && cfg.tile_k > 128)) {
        std::cerr << "tile_k must be positive; shared_tiled requires tile_k <= 128\n";
        std::exit(EXIT_FAILURE);
    }
    if (cfg.warmup < 0 || cfg.repeats <= 0 || cfg.verify_samples < 0) {
        std::cerr << "warmup >= 0, repeats > 0, verify_samples >= 0 are required\n";
        std::exit(EXIT_FAILURE);
    }
}

static size_t shared_bytes(const Config &cfg) {
    return static_cast<size_t>(cfg.block_y * cfg.tile_k + cfg.tile_k * cfg.block_x) *
           sizeof(float);
}

static void check_shared_memory_limit(const Config &cfg) {
    if (cfg.kernel != Kernel::SharedTiled) {
        return;
    }

    cudaDeviceProp prop{};
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    if (shared_bytes(cfg) > prop.sharedMemPerBlock) {
        std::cerr << "Requested shared memory " << shared_bytes(cfg)
                  << " exceeds device limit " << prop.sharedMemPerBlock << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static void fill_random(std::vector<float> &data, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &x : data) {
        x = dist(rng);
    }
}

static std::vector<float> transpose_b(const std::vector<float> &b, int n, int k) {
    std::vector<float> bt(static_cast<size_t>(n) * k);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col) {
            bt[static_cast<size_t>(col) * n + row] =
                b[static_cast<size_t>(row) * k + col];
        }
    }
    return bt;
}

static void launch_kernel(const Config &cfg,
                          const float *d_a,
                          const float *d_b,
                          const float *d_bt,
                          float *d_c) {
    if (cfg.kernel == Kernel::Global1D) {
        int blocks = (cfg.m * cfg.k + cfg.block_x - 1) / cfg.block_x;
        matmul_global1d<<<blocks, cfg.block_x>>>(d_a, d_b, d_c, cfg.m, cfg.n, cfg.k);
        return;
    }

    dim3 block(cfg.block_x, cfg.block_y);
    dim3 grid((cfg.k + block.x - 1) / block.x,
              (cfg.m + block.y - 1) / block.y);

    if (cfg.kernel == Kernel::Global2D) {
        matmul_global2d<<<grid, block>>>(d_a, d_b, d_c, cfg.m, cfg.n, cfg.k);
    } else if (cfg.kernel == Kernel::Global2DBTrans) {
        matmul_global2d_btrans<<<grid, block>>>(d_a, d_bt, d_c, cfg.m, cfg.n, cfg.k);
    } else {
        matmul_shared_tiled<<<grid, block, shared_bytes(cfg)>>>(
            d_a, d_b, d_c, cfg.m, cfg.n, cfg.k, cfg.tile_k);
    }
}

static Timing benchmark(const Config &cfg,
                        const float *d_a,
                        const float *d_b,
                        const float *d_bt,
                        float *d_c) {
    // Warmup removes one-time startup effects from the measured iterations.
    for (int i = 0; i < cfg.warmup; ++i) {
        launch_kernel(cfg, d_a, d_b, d_bt, d_c);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    Timing timing;
    timing.min_ms = std::numeric_limits<float>::max();
    for (int i = 0; i < cfg.repeats; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        launch_kernel(cfg, d_a, d_b, d_bt, d_c);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaGetLastError());

        float elapsed = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        timing.avg_ms += elapsed;
        timing.min_ms = std::min(timing.min_ms, elapsed);
        timing.max_ms = std::max(timing.max_ms, elapsed);
    }
    timing.avg_ms /= static_cast<double>(cfg.repeats);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return timing;
}

static double reference_at(const std::vector<float> &a,
                           const std::vector<float> &b,
                           const Config &cfg,
                           int row,
                           int col) {
    double sum = 0.0;
    for (int p = 0; p < cfg.n; ++p) {
        sum += static_cast<double>(a[static_cast<size_t>(row) * cfg.n + p]) *
               static_cast<double>(b[static_cast<size_t>(p) * cfg.k + col]);
    }
    return sum;
}

static std::vector<size_t> sample_output_indices(const Config &cfg) {
    int total = cfg.m * cfg.k;
    int requested = std::min(cfg.verify_samples, total);
    std::vector<size_t> indices;
    if (requested == 0) {
        return indices;
    }

    indices.push_back(0);
    if (requested > 1) {
        indices.push_back(static_cast<size_t>(total / 2));
    }
    if (requested > 2) {
        indices.push_back(static_cast<size_t>(total - 1));
    }

    std::mt19937 rng(kSeedA + 17);
    std::uniform_int_distribution<int> dist(0, total - 1);
    while (static_cast<int>(indices.size()) < requested) {
        indices.push_back(static_cast<size_t>(dist(rng)));
    }
    return indices;
}

static Verification verify_result(const std::vector<float> &a,
                                  const std::vector<float> &b,
                                  const std::vector<float> &c,
                                  const Config &cfg) {
    // The CPU checks only a fixed sample, so large experiments stay fast.
    Verification result;
    for (size_t idx : sample_output_indices(cfg)) {
        int row = static_cast<int>(idx / cfg.k);
        int col = static_cast<int>(idx % cfg.k);
        double ref = reference_at(a, b, cfg, row, col);
        result.max_abs_error =
            std::max(result.max_abs_error, std::abs(ref - static_cast<double>(c[idx])));
    }

    double tolerance = std::max(1.0e-2, 1.0e-5 * static_cast<double>(cfg.n));
    result.ok = result.max_abs_error <= tolerance;
    return result;
}

static double checksum(const std::vector<float> &data) {
    double sum = 0.0;
    for (float x : data) {
        sum += static_cast<double>(x);
    }
    return sum;
}

static void print_csv_header() {
    std::cout
        << "kernel,m,n,k,block_x,block_y,tile_k,warmup,repeats,"
        << "avg_ms,min_ms,max_ms,gflops,verified,max_abs_error,checksum\n";
}

static void print_csv_row(const Config &cfg,
                          const Timing &timing,
                          const Verification &verification,
                          double output_checksum) {
    double gflops =
        (2.0 * static_cast<double>(cfg.m) * cfg.n * cfg.k) / (timing.avg_ms * 1.0e6);

    std::cout << cfg.kernel_name << ','
              << cfg.m << ','
              << cfg.n << ','
              << cfg.k << ','
              << cfg.block_x << ','
              << cfg.block_y << ','
              << cfg.tile_k << ','
              << cfg.warmup << ','
              << cfg.repeats << ','
              << std::fixed << std::setprecision(6)
              << timing.avg_ms << ','
              << timing.min_ms << ','
              << timing.max_ms << ','
              << gflops << ','
              << (verification.ok ? "yes" : "no") << ','
              << verification.max_abs_error << ','
              << output_checksum << '\n';
}

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);
    validate_config(cfg);
    check_shared_memory_limit(cfg);

    size_t a_count = static_cast<size_t>(cfg.m) * cfg.n;
    size_t b_count = static_cast<size_t>(cfg.n) * cfg.k;
    size_t c_count = static_cast<size_t>(cfg.m) * cfg.k;

    std::vector<float> h_a(a_count);
    std::vector<float> h_b(b_count);
    std::vector<float> h_c(c_count);
    fill_random(h_a, kSeedA);
    fill_random(h_b, kSeedB);

    std::vector<float> h_bt;
    if (cfg.kernel == Kernel::Global2DBTrans) {
        h_bt = transpose_b(h_b, cfg.n, cfg.k);
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_bt = nullptr;
    float *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, a_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, b_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, c_count * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));

    if (!h_bt.empty()) {
        CHECK_CUDA(cudaMalloc(&d_bt, b_count * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_bt, h_bt.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));
    }

    Timing timing = benchmark(cfg, d_a, d_b, d_bt, d_c);
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, c_count * sizeof(float), cudaMemcpyDeviceToHost));

    Verification verification = verify_result(h_a, h_b, h_c, cfg);
    print_csv_header();
    print_csv_row(cfg, timing, verification, checksum(h_c));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    if (d_bt != nullptr) {
        CHECK_CUDA(cudaFree(d_bt));
    }
    return verification.ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
