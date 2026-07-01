#include "cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInputChannels = 3;
constexpr int kKernelSize = 3;
constexpr int kKernelElements = kInputChannels * kKernelSize * kKernelSize;
constexpr int kMaxConstWeights = 512 * kKernelElements;

__constant__ float g_const_weights[kMaxConstWeights];

struct Options {
    int size = 256;
    int stride = 1;
    int out_channels = 1;
    int warmup = 5;
    int iters = 20;
    bool csv_header = false;
};

// All shape-derived values live here, so the kernels and runners do not carry
// long argument lists. Padding is fixed to 1 for the 3x3 CNN-style convolution.
struct ConvShape {
    int height = 0;
    int width = 0;
    int stride = 1;
    int pad = 1;
    int out_channels = 1;
    int out_h = 0;
    int out_w = 0;
    size_t input_elems = 0;
    size_t weight_elems = 0;
    size_t output_elems = 0;
    size_t column_elems = 0;
    double flops = 0.0;
};

struct TimingStats {
    double mean_ms = 0.0;
    double std_ms = 0.0;
    double min_ms = 0.0;
};

struct MethodResult {
    std::string method;
    TimingStats total;
    TimingStats im2col;
    TimingStats gemm;
    double workspace_mb = 0.0;
    double gflops = 0.0;
    double checksum = 0.0;
    float max_abs_error = 0.0f;
    std::vector<float> samples;
    std::string algorithm = "-";
};

class CudnnConvDescriptors {
  public:
    explicit CudnnConvDescriptors(const ConvShape &shape) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(input, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                                               kInputChannels, shape.height, shape.width));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                               shape.out_channels, kInputChannels, kKernelSize,
                                               kKernelSize));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv, shape.pad, shape.pad, shape.stride,
                                                    shape.stride, 1, 1, CUDNN_CROSS_CORRELATION,
                                                    CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                                               shape.out_channels, shape.out_h, shape.out_w));

        int n = 0;
        int c = 0;
        int h = 0;
        int w = 0;
        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv, input, filter, &n, &c, &h, &w));
        if (n != 1 || c != shape.out_channels || h != shape.out_h || w != shape.out_w) {
            throw std::runtime_error("cuDNN output shape differs from CUDA output shape");
        }
    }

    ~CudnnConvDescriptors() {
        if (conv != nullptr) {
            cudnnDestroyConvolutionDescriptor(conv);
        }
        if (filter != nullptr) {
            cudnnDestroyFilterDescriptor(filter);
        }
        if (output != nullptr) {
            cudnnDestroyTensorDescriptor(output);
        }
        if (input != nullptr) {
            cudnnDestroyTensorDescriptor(input);
        }
    }

    cudnnTensorDescriptor_t input = nullptr;
    cudnnTensorDescriptor_t output = nullptr;
    cudnnFilterDescriptor_t filter = nullptr;
    cudnnConvolutionDescriptor_t conv = nullptr;
};

__global__ void direct_conv_kernel(ConvShape shape,
                                   const float *__restrict__ input,
                                   float *__restrict__ output) {
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;

    // One output element is one dot product between a 3x3x3 input patch and one filter.
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < shape.output_elems; idx += step) {
        const int ow = static_cast<int>(idx % shape.out_w);
        const int oh = static_cast<int>((idx / shape.out_w) % shape.out_h);
        const int oc = static_cast<int>(idx / (static_cast<size_t>(shape.out_h) * shape.out_w));
        float sum = 0.0f;

#pragma unroll
        for (int ic = 0; ic < kInputChannels; ++ic) {
#pragma unroll
            for (int kh = 0; kh < kKernelSize; ++kh) {
                const int ih = oh * shape.stride + kh - shape.pad;
                if (ih < 0 || ih >= shape.height) {
                    continue;
                }
#pragma unroll
                for (int kw = 0; kw < kKernelSize; ++kw) {
                    const int iw = ow * shape.stride + kw - shape.pad;
                    if (iw < 0 || iw >= shape.width) {
                        continue;
                    }
                    const int input_idx = (ic * shape.height + ih) * shape.width + iw;
                    const int weight_idx =
                        ((oc * kInputChannels + ic) * kKernelSize + kh) * kKernelSize + kw;
                    sum = fmaf(input[input_idx], g_const_weights[weight_idx], sum);
                }
            }
        }
        output[idx] = sum;
    }
}

__global__ void im2col_kernel(ConvShape shape,
                              const float *__restrict__ input,
                              float *__restrict__ columns) {
    const size_t n_cols = static_cast<size_t>(shape.out_h) * shape.out_w;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;

    // columns is [K*C, out_h*out_w]. Each thread writes one scalar from one receptive field.
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < shape.column_elems; idx += step) {
        const int kernel_pos = static_cast<int>(idx / n_cols);
        const int out_pos = static_cast<int>(idx % n_cols);
        const int ow = out_pos % shape.out_w;
        const int oh = out_pos / shape.out_w;
        const int ic = kernel_pos / (kKernelSize * kKernelSize);
        const int rem = kernel_pos % (kKernelSize * kKernelSize);
        const int kh = rem / kKernelSize;
        const int kw = rem % kKernelSize;
        const int ih = oh * shape.stride + kh - shape.pad;
        const int iw = ow * shape.stride + kw - shape.pad;

        float value = 0.0f;
        if (ih >= 0 && ih < shape.height && iw >= 0 && iw < shape.width) {
            value = input[(ic * shape.height + ih) * shape.width + iw];
        }
        columns[idx] = value;
    }
}

__global__ void checksum_kernel(const float *__restrict__ data, size_t n, double *__restrict__ out) {
    __shared__ double shared[256];
    double local = 0.0;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
        local += static_cast<double>(data[i]);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out, shared[0]);
    }
}

__global__ void max_abs_diff_kernel(const float *__restrict__ a,
                                    const float *__restrict__ b,
                                    size_t n,
                                    unsigned int *__restrict__ out_bits) {
    __shared__ float shared[256];
    float local = 0.0f;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
        local = fmaxf(local, fabsf(a[i] - b[i]));
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(out_bits, __float_as_uint(shared[0]));
    }
}

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

int launch_blocks(size_t work_items) {
    constexpr int threads = 256;
    const size_t capped = std::min<size_t>(work_items, 1u << 30);
    const size_t blocks = ceil_div(static_cast<int>(capped), threads);
    return static_cast<int>(std::min<size_t>(std::max<size_t>(blocks, 1), 131072));
}

ConvShape make_shape(const Options &opt) {
    ConvShape shape;
    shape.height = opt.size;
    shape.width = opt.size;
    shape.stride = opt.stride;
    shape.out_channels = opt.out_channels;
    shape.out_h = (shape.height + 2 * shape.pad - kKernelSize) / shape.stride + 1;
    shape.out_w = (shape.width + 2 * shape.pad - kKernelSize) / shape.stride + 1;
    shape.input_elems = static_cast<size_t>(kInputChannels) * shape.height * shape.width;
    shape.weight_elems = static_cast<size_t>(shape.out_channels) * kKernelElements;
    shape.output_elems = static_cast<size_t>(shape.out_channels) * shape.out_h * shape.out_w;
    shape.column_elems = static_cast<size_t>(kKernelElements) * shape.out_h * shape.out_w;
    shape.flops = static_cast<double>(shape.output_elems) * kKernelElements * 2.0;
    return shape;
}

TimingStats summarize(const std::vector<float> &times) {
    TimingStats stats;
    if (times.empty()) {
        return stats;
    }
    stats.mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    stats.min_ms = *std::min_element(times.begin(), times.end());
    double var = 0.0;
    for (float t : times) {
        const double d = t - stats.mean_ms;
        var += d * d;
    }
    stats.std_ms = std::sqrt(var / times.size());
    return stats;
}

template <typename Func>
TimingStats benchmark(int warmup, int iters, Func &&run_once) {
    for (int i = 0; i < warmup; ++i) {
        run_once();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvents events;
    std::vector<float> times;
    times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        events.start();
        run_once();
        times.push_back(events.stop());
        CUDA_CHECK(cudaGetLastError());
    }
    return summarize(times);
}

void fill_data(std::vector<float> &data, uint32_t seed, float scale) {
    uint32_t state = seed;
    for (float &x : data) {
        state = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) / 16777215.0f;
        x = (u - 0.5f) * scale;
    }
}

double checksum_device(const float *data, size_t n) {
    DeviceBuffer<double> device_sum(1);
    CUDA_CHECK(cudaMemset(device_sum.get(), 0, sizeof(double)));
    checksum_kernel<<<launch_blocks(n), 256>>>(data, n, device_sum.get());
    CUDA_CHECK(cudaGetLastError());

    double host_sum = 0.0;
    CUDA_CHECK(cudaMemcpy(&host_sum, device_sum.get(), sizeof(double), cudaMemcpyDeviceToHost));
    return host_sum;
}

float max_abs_diff_device(const float *a, const float *b, size_t n) {
    DeviceBuffer<unsigned int> device_bits(1);
    CUDA_CHECK(cudaMemset(device_bits.get(), 0, sizeof(unsigned int)));
    max_abs_diff_kernel<<<launch_blocks(n), 256>>>(a, b, n, device_bits.get());
    CUDA_CHECK(cudaGetLastError());

    unsigned int host_bits = 0;
    CUDA_CHECK(cudaMemcpy(&host_bits, device_bits.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    float value = 0.0f;
    std::memcpy(&value, &host_bits, sizeof(float));
    return value;
}

std::vector<float> copy_samples(const float *data, size_t n) {
    std::vector<float> samples(4, 0.0f);
    const size_t count = std::min<size_t>(samples.size(), n);
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(samples.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return samples;
}

void finish_result(MethodResult &result,
                   const ConvShape &shape,
                   const float *output,
                   const float *reference = nullptr) {
    result.gflops = shape.flops / (result.total.mean_ms * 1.0e6);
    result.checksum = checksum_device(output, shape.output_elems);
    result.samples = copy_samples(output, shape.output_elems);
    if (reference != nullptr && reference != output) {
        result.max_abs_error = max_abs_diff_device(reference, output, shape.output_elems);
    }
}

MethodResult run_direct(const Options &opt,
                        const ConvShape &shape,
                        const float *input,
                        float *reference_output) {
    MethodResult result;
    result.method = "direct_cuda";

    const int blocks = launch_blocks(shape.output_elems);
    auto run_once = [&] {
        direct_conv_kernel<<<blocks, 256>>>(shape, input, reference_output);
    };

    result.total = benchmark(opt.warmup, opt.iters, run_once);
    finish_result(result, shape, reference_output);
    return result;
}

MethodResult run_im2col_gemm(const Options &opt,
                             const ConvShape &shape,
                             const float *input,
                             const float *weight,
                             const float *reference,
                             cublasHandle_t cublas) {
    MethodResult result;
    result.method = "im2col_cublas";

    DeviceBuffer<float> columns(shape.column_elems);
    DeviceBuffer<float> output(shape.output_elems);
    result.workspace_mb = static_cast<double>(shape.column_elems * sizeof(float)) / (1024.0 * 1024.0);

    const int n_cols = shape.out_h * shape.out_w;
    const int blocks = launch_blocks(shape.column_elems);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto run_im2col = [&] {
        im2col_kernel<<<blocks, 256>>>(shape, input, columns.get());
    };
    auto run_gemm = [&] {
        // cuBLAS is column-major. These dimensions make C's memory layout match
        // row-major output [out_channels][out_h*out_w] without a separate transpose.
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n_cols, shape.out_channels,
                                 kKernelElements, &alpha, columns.get(), n_cols, weight,
                                 kKernelElements, &beta, output.get(), n_cols));
    };

    for (int i = 0; i < opt.warmup; ++i) {
        run_im2col();
        run_gemm();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvents total_events;
    CudaEvents split_events;
    std::vector<float> total_times;
    std::vector<float> im2col_times;
    std::vector<float> gemm_times;
    total_times.reserve(opt.iters);
    im2col_times.reserve(opt.iters);
    gemm_times.reserve(opt.iters);

    for (int i = 0; i < opt.iters; ++i) {
        total_events.start();
        split_events.start();
        run_im2col();
        im2col_times.push_back(split_events.stop());
        split_events.start();
        run_gemm();
        gemm_times.push_back(split_events.stop());
        total_times.push_back(total_events.stop());
        CUDA_CHECK(cudaGetLastError());
    }

    result.total = summarize(total_times);
    result.im2col = summarize(im2col_times);
    result.gemm = summarize(gemm_times);
    finish_result(result, shape, output.get(), reference);
    return result;
}

std::string algo_name(cudnnConvolutionFwdAlgo_t algo) {
    switch (algo) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
        return "IMPLICIT_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
        return "IMPLICIT_PRECOMP_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
        return "GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
        return "DIRECT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
        return "FFT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
        return "FFT_TILING";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
        return "WINOGRAD";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
        return "WINOGRAD_NONFUSED";
    default:
        return "UNKNOWN";
    }
}

cudnnConvolutionFwdAlgo_t choose_cudnn_algorithm(cudnnHandle_t cudnn,
                                                 const CudnnConvDescriptors &desc,
                                                 size_t *workspace_bytes) {
    std::vector<cudnnConvolutionFwdAlgo_t> candidates;

    int max_count = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &max_count));
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf(max_count);
    int returned_count = 0;
    const cudnnStatus_t find_status =
        cudnnFindConvolutionForwardAlgorithm(cudnn, desc.input, desc.filter, desc.conv, desc.output,
                                             max_count, &returned_count, perf.data());
    if (find_status == CUDNN_STATUS_SUCCESS) {
        for (int i = 0; i < returned_count; ++i) {
            if (perf[i].status == CUDNN_STATUS_SUCCESS) {
                candidates.push_back(perf[i].algo);
            }
        }
    }

    if (candidates.empty()) {
        candidates = {
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        };
    }

    for (cudnnConvolutionFwdAlgo_t algo : candidates) {
        size_t bytes = 0;
        cudnnStatus_t status =
            cudnnGetConvolutionForwardWorkspaceSize(cudnn, desc.input, desc.filter, desc.conv,
                                                    desc.output, algo, &bytes);
        if (status != CUDNN_STATUS_SUCCESS) {
            continue;
        }

        DeviceBuffer<unsigned char> workspace_probe;
        try {
            workspace_probe.reset(bytes);
        } catch (const std::runtime_error &) {
            cudaGetLastError();
            continue;
        }
        *workspace_bytes = bytes;
        return algo;
    }

    *workspace_bytes = 0;
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
}

MethodResult run_cudnn(const Options &opt,
                       const ConvShape &shape,
                       const float *input,
                       const float *weight,
                       const float *reference,
                       cudnnHandle_t cudnn) {
    MethodResult result;
    result.method = "cudnn";

    CudnnConvDescriptors desc(shape);
    DeviceBuffer<float> output(shape.output_elems);

    size_t workspace_bytes = 0;
    cudnnConvolutionFwdAlgo_t algo = choose_cudnn_algorithm(cudnn, desc, &workspace_bytes);
    DeviceBuffer<unsigned char> workspace(workspace_bytes);

    result.algorithm = algo_name(algo);
    result.workspace_mb = static_cast<double>(workspace_bytes) / (1024.0 * 1024.0);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto run_once = [&] {
        CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, desc.input, input, desc.filter, weight,
                                            desc.conv, algo, workspace.get(), workspace_bytes,
                                            &beta, desc.output, output.get()));
    };

    result.total = benchmark(opt.warmup, opt.iters, run_once);
    finish_result(result, shape, output.get(), reference);
    return result;
}

void print_csv_header() {
    std::cout << "method,input_size,input_channels,kernel,out_channels,stride,padding,out_h,out_w,"
              << "warmup,iterations,mean_ms,std_ms,min_ms,gflops,workspace_mb,im2col_ms,gemm_ms,"
              << "max_abs_error,checksum,sample0,sample1,sample2,sample3,algorithm,gpu\n";
}

std::string csv_escape(const std::string &s) {
    if (s.find_first_of(",\"\n") == std::string::npos) {
        return s;
    }
    std::string out = "\"";
    for (char c : s) {
        out += (c == '"') ? "\"\"" : std::string(1, c);
    }
    out += '"';
    return out;
}

void print_result(const MethodResult &result,
                  const Options &opt,
                  const ConvShape &shape,
                  const std::string &gpu) {
    std::cout << std::fixed << std::setprecision(6)
              << result.method << ','
              << opt.size << ','
              << kInputChannels << ','
              << kKernelSize << ','
              << opt.out_channels << ','
              << opt.stride << ','
              << shape.pad << ','
              << shape.out_h << ','
              << shape.out_w << ','
              << opt.warmup << ','
              << opt.iters << ','
              << result.total.mean_ms << ','
              << result.total.std_ms << ','
              << result.total.min_ms << ','
              << result.gflops << ','
              << result.workspace_mb << ','
              << result.im2col.mean_ms << ','
              << result.gemm.mean_ms << ','
              << result.max_abs_error << ','
              << result.checksum << ','
              << result.samples[0] << ','
              << result.samples[1] << ','
              << result.samples[2] << ','
              << result.samples[3] << ','
              << csv_escape(result.algorithm) << ','
              << csv_escape(gpu) << '\n';
}

Options parse_options(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--size") {
            opt.size = std::atoi(require_value("--size"));
        } else if (arg == "--stride") {
            opt.stride = std::atoi(require_value("--stride"));
        } else if (arg == "--out-channels") {
            opt.out_channels = std::atoi(require_value("--out-channels"));
        } else if (arg == "--warmup") {
            opt.warmup = std::atoi(require_value("--warmup"));
        } else if (arg == "--iters") {
            opt.iters = std::atoi(require_value("--iters"));
        } else if (arg == "--csv-header") {
            opt.csv_header = true;
        } else if (arg == "--help") {
            std::cout << "Usage: conv_benchmark [--size N] [--stride 1|2|3] "
                      << "[--out-channels K] [--warmup N] [--iters N] [--csv-header]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (opt.size <= 0 || opt.stride <= 0 || opt.out_channels <= 0 || opt.warmup < 0 ||
        opt.iters <= 0) {
        throw std::runtime_error("invalid numeric option");
    }
    if (opt.out_channels * kKernelElements > kMaxConstWeights) {
        throw std::runtime_error("out_channels is too large for constant-memory direct kernel");
    }
    return opt;
}

std::string current_gpu_name() {
    cudaDeviceProp prop{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.name;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options opt = parse_options(argc, argv);
        if (opt.csv_header) {
            print_csv_header();
            return 0;
        }

        CUDA_CHECK(cudaSetDevice(0));
        const std::string gpu = current_gpu_name();
        const ConvShape shape = make_shape(opt);

        std::vector<float> host_input(shape.input_elems);
        std::vector<float> host_weight(shape.weight_elems);
        fill_data(host_input, 1234u + static_cast<uint32_t>(opt.size), 2.0f);
        fill_data(host_weight, 5678u + static_cast<uint32_t>(opt.out_channels), 0.25f);

        DeviceBuffer<float> input(shape.input_elems);
        DeviceBuffer<float> weight(shape.weight_elems);
        DeviceBuffer<float> reference(shape.output_elems);
        CUDA_CHECK(cudaMemcpy(input.get(), host_input.data(), shape.input_elems * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(weight.get(), host_weight.data(), shape.weight_elems * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(g_const_weights, host_weight.data(),
                                      shape.weight_elems * sizeof(float)));

        CublasHandle cublas;
        CudnnHandle cudnn;

        std::vector<MethodResult> results;
        results.push_back(run_direct(opt, shape, input.get(), reference.get()));
        results.push_back(run_im2col_gemm(opt, shape, input.get(), weight.get(), reference.get(),
                                          cublas));
        results.push_back(run_cudnn(opt, shape, input.get(), weight.get(), reference.get(), cudnn));

        for (const MethodResult &result : results) {
            print_result(result, opt, shape, gpu);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "conv_benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
