#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << ": " << cudaGetErrorString(err__) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

namespace {

enum class Variant {
    NaiveWriteStrided,
    NaiveReadStrided,
    TiledNoPad,
    TiledPad
};

struct Config {
    int tile = 32;
    int block_rows = 8;
};

struct BenchRow {
    Variant variant;
    int n = 0;
    Config config;
    int repeat = 0;
    float time_ms = 0.0f;
    double bandwidth_gbs = 0.0;
};

constexpr int kMaxThreadsPerBlock = 1024;

__global__ void init_matrix_kernel(float *matrix, int n) {
    std::uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint64_t total = static_cast<std::uint64_t>(n) * n;
    std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

    for (std::uint64_t i = idx; i < total; i += stride) {
        std::uint64_t row = i / n;
        std::uint64_t col = i - row * n;
        std::uint32_t v = static_cast<std::uint32_t>((row * 1315423911ULL) ^
                                                     (col * 2654435761ULL));
        matrix[i] = static_cast<float>(v & 0xffffu) * 0.25f;
    }
}

__global__ void transpose_naive_write_strided(const float *__restrict__ in,
                                              float *__restrict__ out,
                                              int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n) {
        out[x * n + y] = in[y * n + x];
    }
}

__global__ void transpose_naive_read_strided(const float *__restrict__ in,
                                             float *__restrict__ out,
                                             int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n) {
        out[y * n + x] = in[x * n + y];
    }
}

template <int TILE_DIM, int BLOCK_ROWS, int PAD>
__global__ void transpose_tiled_shared(const float *__restrict__ in,
                                       float *__restrict__ out,
                                       int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM + PAD];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int yy = y + j;
        if (x < n && yy < n) {
            tile[threadIdx.y + j][threadIdx.x] = in[yy * n + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int yy = y + j;
        if (x < n && yy < n) {
            out[yy * n + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

std::string variant_name(Variant variant) {
    switch (variant) {
        case Variant::NaiveWriteStrided:
            return "naive_write_strided";
        case Variant::NaiveReadStrided:
            return "naive_read_strided";
        case Variant::TiledNoPad:
            return "tiled_nopad";
        case Variant::TiledPad:
            return "tiled_pad";
    }
    return "unknown";
}

bool parse_variant(const std::string &name, Variant *variant) {
    if (name == "naive_write_strided" || name == "naive_write") {
        *variant = Variant::NaiveWriteStrided;
        return true;
    }
    if (name == "naive_read_strided" || name == "naive_read") {
        *variant = Variant::NaiveReadStrided;
        return true;
    }
    if (name == "tiled_nopad" || name == "tiled") {
        *variant = Variant::TiledNoPad;
        return true;
    }
    if (name == "tiled_pad" || name == "shared") {
        *variant = Variant::TiledPad;
        return true;
    }
    return false;
}

std::vector<std::string> split(const std::string &text, char delim) {
    std::vector<std::string> parts;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            parts.push_back(item);
        }
    }
    return parts;
}

std::vector<int> parse_sizes(const std::string &text) {
    std::vector<int> sizes;
    for (const std::string &part : split(text, ',')) {
        int n = std::atoi(part.c_str());
        if (n <= 0) {
            std::cerr << "Invalid matrix size: " << part << std::endl;
            std::exit(EXIT_FAILURE);
        }
        sizes.push_back(n);
    }
    return sizes;
}

Config parse_config_one(const std::string &text) {
    std::size_t pos = text.find('x');
    if (pos == std::string::npos) {
        pos = text.find('X');
    }
    if (pos == std::string::npos) {
        std::cerr << "Invalid config '" << text << "', expected TILExROWS.\n";
        std::exit(EXIT_FAILURE);
    }
    Config cfg;
    cfg.tile = std::atoi(text.substr(0, pos).c_str());
    cfg.block_rows = std::atoi(text.substr(pos + 1).c_str());
    if (cfg.tile <= 0 || cfg.block_rows <= 0 ||
        cfg.tile * cfg.block_rows > kMaxThreadsPerBlock) {
        std::cerr << "Invalid config '" << text
                  << "': tile * block_rows must be in (0, 1024].\n";
        std::exit(EXIT_FAILURE);
    }
    return cfg;
}

std::vector<Config> parse_configs(const std::string &text) {
    std::vector<Config> configs;
    for (const std::string &part : split(text, ',')) {
        configs.push_back(parse_config_one(part));
    }
    return configs;
}

std::vector<Variant> parse_variants(const std::string &text) {
    std::vector<Variant> variants;
    for (const std::string &part : split(text, ',')) {
        Variant v;
        if (!parse_variant(part, &v)) {
            std::cerr << "Unknown variant: " << part << std::endl;
            std::exit(EXIT_FAILURE);
        }
        variants.push_back(v);
    }
    return variants;
}

bool supported_tiled_config(Config cfg) {
    if (cfg.tile != 8 && cfg.tile != 16 && cfg.tile != 32 && cfg.tile != 64) {
        return false;
    }
    if (cfg.block_rows != 4 && cfg.block_rows != 8 &&
        cfg.block_rows != 16 && cfg.block_rows != 32) {
        return false;
    }
    return cfg.block_rows <= cfg.tile &&
           cfg.tile % cfg.block_rows == 0 &&
           cfg.tile * cfg.block_rows <= kMaxThreadsPerBlock;
}

bool supported_config(Variant variant, Config cfg) {
    if (cfg.tile * cfg.block_rows > kMaxThreadsPerBlock) {
        return false;
    }
    if (variant == Variant::NaiveWriteStrided ||
        variant == Variant::NaiveReadStrided) {
        return true;
    }
    return supported_tiled_config(cfg);
}

template <int TILE, int ROWS, int PAD>
void launch_tiled_impl(const float *in, float *out, int n, cudaStream_t stream) {
    dim3 block(TILE, ROWS);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    transpose_tiled_shared<TILE, ROWS, PAD><<<grid, block, 0, stream>>>(in, out, n);
}

template <int PAD>
bool launch_tiled_dispatch(Config cfg, const float *in, float *out,
                           int n, cudaStream_t stream) {
    if (cfg.tile == 8 && cfg.block_rows == 4) {
        launch_tiled_impl<8, 4, PAD>(in, out, n, stream);
    } else if (cfg.tile == 8 && cfg.block_rows == 8) {
        launch_tiled_impl<8, 8, PAD>(in, out, n, stream);
    } else if (cfg.tile == 16 && cfg.block_rows == 4) {
        launch_tiled_impl<16, 4, PAD>(in, out, n, stream);
    } else if (cfg.tile == 16 && cfg.block_rows == 8) {
        launch_tiled_impl<16, 8, PAD>(in, out, n, stream);
    } else if (cfg.tile == 16 && cfg.block_rows == 16) {
        launch_tiled_impl<16, 16, PAD>(in, out, n, stream);
    } else if (cfg.tile == 32 && cfg.block_rows == 4) {
        launch_tiled_impl<32, 4, PAD>(in, out, n, stream);
    } else if (cfg.tile == 32 && cfg.block_rows == 8) {
        launch_tiled_impl<32, 8, PAD>(in, out, n, stream);
    } else if (cfg.tile == 32 && cfg.block_rows == 16) {
        launch_tiled_impl<32, 16, PAD>(in, out, n, stream);
    } else if (cfg.tile == 32 && cfg.block_rows == 32) {
        launch_tiled_impl<32, 32, PAD>(in, out, n, stream);
    } else if (cfg.tile == 64 && cfg.block_rows == 4) {
        launch_tiled_impl<64, 4, PAD>(in, out, n, stream);
    } else if (cfg.tile == 64 && cfg.block_rows == 8) {
        launch_tiled_impl<64, 8, PAD>(in, out, n, stream);
    } else if (cfg.tile == 64 && cfg.block_rows == 16) {
        launch_tiled_impl<64, 16, PAD>(in, out, n, stream);
    } else {
        return false;
    }
    return true;
}

void launch_variant(Variant variant, Config cfg, const float *in, float *out,
                    int n, cudaStream_t stream = 0) {
    if (!supported_config(variant, cfg)) {
        std::cerr << "Unsupported config " << cfg.tile << "x" << cfg.block_rows
                  << " for " << variant_name(variant) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (variant == Variant::NaiveWriteStrided) {
        dim3 block(cfg.tile, cfg.block_rows);
        dim3 grid((n + block.x - 1) / block.x,
                  (n + block.y - 1) / block.y);
        transpose_naive_write_strided<<<grid, block, 0, stream>>>(in, out, n);
        return;
    }

    if (variant == Variant::NaiveReadStrided) {
        dim3 block(cfg.tile, cfg.block_rows);
        dim3 grid((n + block.x - 1) / block.x,
                  (n + block.y - 1) / block.y);
        transpose_naive_read_strided<<<grid, block, 0, stream>>>(in, out, n);
        return;
    }

    bool launched = false;
    if (variant == Variant::TiledNoPad) {
        launched = launch_tiled_dispatch<0>(cfg, in, out, n, stream);
    } else if (variant == Variant::TiledPad) {
        launched = launch_tiled_dispatch<1>(cfg, in, out, n, stream);
    }
    if (!launched) {
        std::cerr << "Unsupported tiled config " << cfg.tile << "x"
                  << cfg.block_rows << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void fill_host_matrix(std::vector<float> *matrix, int n) {
    matrix->resize(static_cast<std::size_t>(n) * n);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            std::uint32_t v = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(row) * 1315423911ULL) ^
                (static_cast<std::uint64_t>(col) * 2654435761ULL));
            (*matrix)[static_cast<std::size_t>(row) * n + col] =
                static_cast<float>(v & 0xffffu) * 0.25f;
        }
    }
}

std::vector<float> cpu_transpose(const std::vector<float> &in, int n) {
    std::vector<float> out(static_cast<std::size_t>(n) * n);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            out[static_cast<std::size_t>(col) * n + row] =
                in[static_cast<std::size_t>(row) * n + col];
        }
    }
    return out;
}

bool verify_once(int n, Variant variant, Config cfg, bool verbose) {
    if (!supported_config(variant, cfg)) {
        return true;
    }

    std::vector<float> h_in;
    fill_host_matrix(&h_in, n);
    std::vector<float> h_expected = cpu_transpose(h_in, n);
    std::vector<float> h_out(h_expected.size(), -1.0f);

    float *d_in = nullptr;
    float *d_out = nullptr;
    std::size_t bytes = h_in.size() * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, bytes));

    launch_variant(variant, cfg, d_in, d_out, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    std::size_t bad_index = 0;
    for (std::size_t i = 0; i < h_out.size(); ++i) {
        if (h_out[i] != h_expected[i]) {
            ok = false;
            bad_index = i;
            break;
        }
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    if (verbose) {
        std::cout << (ok ? "[PASS] " : "[FAIL] ")
                  << "n=" << n
                  << " variant=" << variant_name(variant)
                  << " config=" << cfg.tile << "x" << cfg.block_rows;
        if (!ok) {
            int row = static_cast<int>(bad_index / n);
            int col = static_cast<int>(bad_index % n);
            std::cout << " mismatch at (" << row << ", " << col << ")"
                      << " got=" << h_out[bad_index]
                      << " expected=" << h_expected[bad_index];
        }
        std::cout << std::endl;
    }
    return ok;
}

int run_tests() {
    std::vector<int> sizes = {1, 2, 3, 7, 16, 31, 32, 33,
                              64, 127, 255, 512, 1024, 2048};
    std::vector<Config> configs = {
        {8, 4}, {8, 8}, {16, 4}, {16, 8}, {16, 16},
        {32, 4}, {32, 8}, {32, 16}, {32, 32},
        {64, 4}, {64, 8}, {64, 16}
    };
    std::vector<Variant> variants = {
        Variant::NaiveWriteStrided,
        Variant::NaiveReadStrided,
        Variant::TiledNoPad,
        Variant::TiledPad
    };

    int total = 0;
    int failed = 0;
    for (int n : sizes) {
        for (Variant variant : variants) {
            for (Config cfg : configs) {
                if (!supported_config(variant, cfg)) {
                    continue;
                }
                bool ok = verify_once(n, variant, cfg, true);
                ++total;
                if (!ok) {
                    ++failed;
                }
            }
        }
    }

    std::cout << "Correctness summary: " << (total - failed) << "/"
              << total << " passed." << std::endl;
    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

float time_kernel_loop(Variant variant, Config cfg, const float *d_in,
                       float *d_out, int n, int repeat) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        launch_variant(variant, cfg, d_in, d_out, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaGetLastError());
    return elapsed_ms / repeat;
}

BenchRow benchmark_one(Variant variant, Config cfg, const float *d_in,
                       float *d_out, int n, int warmup, int repeat,
                       float min_ms) {
    for (int i = 0; i < warmup; ++i) {
        launch_variant(variant, cfg, d_in, d_out, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int actual_repeat = repeat;
    if (actual_repeat <= 0) {
        int trial_repeat = 3;
        float trial_avg = time_kernel_loop(variant, cfg, d_in, d_out,
                                           n, trial_repeat);
        actual_repeat = static_cast<int>(std::ceil(min_ms / trial_avg));
        actual_repeat = std::max(5, std::min(actual_repeat, 20000));
    }

    float avg_ms = time_kernel_loop(variant, cfg, d_in, d_out, n,
                                    actual_repeat);
    double bytes = 2.0 * static_cast<double>(n) * n * sizeof(float);
    double bandwidth_gbs = bytes / (static_cast<double>(avg_ms) * 1.0e6);

    BenchRow row;
    row.variant = variant;
    row.n = n;
    row.config = cfg;
    row.repeat = actual_repeat;
    row.time_ms = avg_ms;
    row.bandwidth_gbs = bandwidth_gbs;
    return row;
}

void write_csv_header(std::ostream &os) {
    os << "variant,n,tile,block_rows,threads_per_block,repeat,time_ms,bandwidth_gbs\n";
}

void write_csv_row(std::ostream &os, const BenchRow &row) {
    os << variant_name(row.variant) << ","
       << row.n << ","
       << row.config.tile << ","
       << row.config.block_rows << ","
       << row.config.tile * row.config.block_rows << ","
       << row.repeat << ","
       << std::fixed << std::setprecision(6) << row.time_ms << ","
       << std::fixed << std::setprecision(3) << row.bandwidth_gbs << "\n";
}

int run_benchmark(const std::vector<int> &sizes,
                  const std::vector<Config> &configs,
                  const std::vector<Variant> &variants,
                  int warmup, int repeat, float min_ms,
                  const std::string &csv_path) {
    std::ostream *out = &std::cout;
    std::ofstream file;
    if (!csv_path.empty()) {
        file.open(csv_path);
        if (!file) {
            std::cerr << "Failed to open CSV path: " << csv_path << std::endl;
            return EXIT_FAILURE;
        }
        out = &file;
    }
    write_csv_header(*out);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "GPU: " << prop.name << ", compute capability "
              << prop.major << "." << prop.minor << "\n";

    for (int n : sizes) {
        std::size_t elements = static_cast<std::size_t>(n) * n;
        std::size_t bytes = elements * sizeof(float);
        std::cerr << "Benchmark n=" << n << " (" << (2.0 * bytes / (1 << 30))
                  << " GiB device buffers)\n";

        float *d_in = nullptr;
        float *d_out = nullptr;
        CHECK_CUDA(cudaMalloc(&d_in, bytes));
        CHECK_CUDA(cudaMalloc(&d_out, bytes));

        int threads = 256;
        int blocks = std::min<int>(65535, static_cast<int>((elements + threads - 1) / threads));
        init_matrix_kernel<<<blocks, threads>>>(d_in, n);
        CHECK_CUDA(cudaMemset(d_out, 0, bytes));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        for (Variant variant : variants) {
            for (Config cfg : configs) {
                if (!supported_config(variant, cfg)) {
                    continue;
                }
                BenchRow row = benchmark_one(variant, cfg, d_in, d_out, n,
                                             warmup, repeat, min_ms);
                write_csv_row(*out, row);
                out->flush();
                std::cerr << "  " << std::setw(20) << variant_name(variant)
                          << " " << std::setw(5) << cfg.tile << "x"
                          << std::left << std::setw(3) << cfg.block_rows
                          << std::right << " "
                          << std::setw(9) << std::fixed << std::setprecision(3)
                          << row.time_ms << " ms, "
                          << std::setw(9) << std::fixed << std::setprecision(1)
                          << row.bandwidth_gbs << " GB/s\n";
            }
        }

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
    }
    return EXIT_SUCCESS;
}

void print_matrix(const std::vector<float> &matrix, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cout << std::setw(8)
                      << matrix[static_cast<std::size_t>(row) * n + col]
                      << " ";
        }
        std::cout << "\n";
    }
}

int run_demo(int n, Variant variant, Config cfg) {
    if (n <= 0 || n > 16) {
        std::cerr << "Demo mode prints matrices and therefore requires n in [1, 16].\n";
        return EXIT_FAILURE;
    }

    std::vector<float> h_in;
    fill_host_matrix(&h_in, n);
    std::vector<float> h_out(static_cast<std::size_t>(n) * n, 0.0f);
    std::size_t bytes = h_in.size() * sizeof(float);

    float *d_in = nullptr;
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    launch_variant(variant, cfg, d_in, d_out, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Input matrix A:\n";
    print_matrix(h_in, n);
    std::cout << "\nTransposed matrix A^T by " << variant_name(variant)
              << " (" << cfg.tile << "x" << cfg.block_rows << "):\n";
    print_matrix(h_out, n);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    return EXIT_SUCCESS;
}

void usage(const char *prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " --test\n"
        << "  " << prog << " --demo --n 8 --variant tiled_pad --config 32x8\n"
        << "  " << prog << " --benchmark --sizes 512,1024,2048 --configs 16x16,32x8 --variants tiled_pad,naive_write_strided --csv results/out.csv\n"
        << "\nOptions:\n"
        << "  --sizes LIST       comma-separated matrix sizes\n"
        << "  --configs LIST     comma-separated TILExROWS block configs\n"
        << "  --variants LIST    naive_write_strided, naive_read_strided, tiled_nopad, tiled_pad\n"
        << "  --warmup N         warmup launches per benchmark row\n"
        << "  --repeat N         timed launches per benchmark row; <=0 enables auto-repeat\n"
        << "  --min-ms MS        target measured time when auto-repeat is enabled\n"
        << "  --csv PATH         write benchmark CSV to PATH\n";
}

} // namespace

int main(int argc, char **argv) {
    bool mode_test = false;
    bool mode_demo = false;
    bool mode_benchmark = false;
    int demo_n = 8;
    Variant demo_variant = Variant::TiledPad;
    Config demo_config{32, 8};
    std::vector<int> sizes = {512, 1024, 1536, 2048};
    std::vector<Config> configs = {
        {8, 4}, {8, 8}, {16, 8}, {16, 16},
        {32, 8}, {32, 16}, {32, 32}, {64, 8}, {64, 16}
    };
    std::vector<Variant> variants = {
        Variant::NaiveWriteStrided,
        Variant::NaiveReadStrided,
        Variant::TiledNoPad,
        Variant::TiledPad
    };
    int warmup = 5;
    int repeat = -1;
    float min_ms = 250.0f;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if (arg == "--test") {
            mode_test = true;
        } else if (arg == "--demo") {
            mode_demo = true;
        } else if (arg == "--benchmark") {
            mode_benchmark = true;
        } else if (arg == "--n") {
            demo_n = std::atoi(require_value("--n"));
        } else if (arg == "--variant") {
            if (!parse_variant(require_value("--variant"), &demo_variant)) {
                std::cerr << "Unknown variant for demo.\n";
                return EXIT_FAILURE;
            }
        } else if (arg == "--config") {
            demo_config = parse_config_one(require_value("--config"));
        } else if (arg == "--sizes") {
            sizes = parse_sizes(require_value("--sizes"));
        } else if (arg == "--configs") {
            configs = parse_configs(require_value("--configs"));
        } else if (arg == "--variants") {
            variants = parse_variants(require_value("--variants"));
        } else if (arg == "--warmup") {
            warmup = std::atoi(require_value("--warmup"));
        } else if (arg == "--repeat") {
            repeat = std::atoi(require_value("--repeat"));
        } else if (arg == "--min-ms") {
            min_ms = std::atof(require_value("--min-ms"));
        } else if (arg == "--csv") {
            csv_path = require_value("--csv");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    int modes = static_cast<int>(mode_test) + static_cast<int>(mode_demo) +
                static_cast<int>(mode_benchmark);
    if (modes != 1) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (mode_test) {
        return run_tests();
    }
    if (mode_demo) {
        return run_demo(demo_n, demo_variant, demo_config);
    }
    return run_benchmark(sizes, configs, variants, warmup, repeat,
                         min_ms, csv_path);
}
