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
constexpr int kTile = 16;
constexpr int kChannelTile = 8;

__constant__ float g_const_weights[kMaxConstWeights];

struct Options {
    int size = 256;
    int stride = 1;
    int out_channels = 1;
    int warmup = 5;
    int iters = 20;
    bool csv_header = false;
};

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
    std::string optimization;
    TimingStats total;
    TimingStats im2col;
    TimingStats gemm;
    double workspace_mb = 0.0;
    double gflops = 0.0;
    double checksum = 0.0;
    float max_abs_error = 0.0f;
    std::string notes = "-";
};

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

__global__ void direct_conv_kernel(ConvShape shape,
                                   const float *__restrict__ input,
                                   float *__restrict__ output) {
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
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

__global__ void shared_tile_kernel(ConvShape shape,
                                   const float *__restrict__ input,
                                   float *__restrict__ output) {
    extern __shared__ float tile[];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int oh = blockIdx.y * blockDim.y + ty;
    const int ow = blockIdx.x * blockDim.x + tx;
    const int oc = blockIdx.z;

    const int tile_h = (blockDim.y - 1) * shape.stride + kKernelSize;
    const int tile_w = (blockDim.x - 1) * shape.stride + kKernelSize;
    const int tile_plane = tile_h * tile_w;
    const int total = kInputChannels * tile_plane;
    const int tid = ty * blockDim.x + tx;
    const int threads = blockDim.x * blockDim.y;
    const int input_y0 = blockIdx.y * blockDim.y * shape.stride - shape.pad;
    const int input_x0 = blockIdx.x * blockDim.x * shape.stride - shape.pad;

    for (int i = tid; i < total; i += threads) {
        const int ic = i / tile_plane;
        const int rem = i % tile_plane;
        const int local_y = rem / tile_w;
        const int local_x = rem % tile_w;
        const int iy = input_y0 + local_y;
        const int ix = input_x0 + local_x;
        float v = 0.0f;
        if (iy >= 0 && iy < shape.height && ix >= 0 && ix < shape.width) {
            v = input[(ic * shape.height + iy) * shape.width + ix];
        }
        tile[i] = v;
    }
    __syncthreads();

    if (oh >= shape.out_h || ow >= shape.out_w) {
        return;
    }

    float sum = 0.0f;
#pragma unroll
    for (int ic = 0; ic < kInputChannels; ++ic) {
#pragma unroll
        for (int kh = 0; kh < kKernelSize; ++kh) {
#pragma unroll
            for (int kw = 0; kw < kKernelSize; ++kw) {
                const int local_y = ty * shape.stride + kh;
                const int local_x = tx * shape.stride + kw;
                const int weight_idx =
                    ((oc * kInputChannels + ic) * kKernelSize + kh) * kKernelSize + kw;
                sum = fmaf(tile[ic * tile_plane + local_y * tile_w + local_x],
                           g_const_weights[weight_idx], sum);
            }
        }
    }
    output[(static_cast<size_t>(oc) * shape.out_h + oh) * shape.out_w + ow] = sum;
}

__global__ void register_block2_kernel(ConvShape shape,
                                       const float *__restrict__ input,
                                       float *__restrict__ output) {
    const int pair_w = (shape.out_w + 1) / 2;
    const size_t pairs_per_channel = static_cast<size_t>(shape.out_h) * pair_w;
    const size_t total_pairs = static_cast<size_t>(shape.out_channels) * pairs_per_channel;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_pairs; idx += step) {
        const int oc = static_cast<int>(idx / pairs_per_channel);
        const int pair_pos = static_cast<int>(idx % pairs_per_channel);
        const int oh = pair_pos / pair_w;
        const int ow0 = (pair_pos % pair_w) * 2;
        const int ow1 = ow0 + 1;
        float sum0 = 0.0f;
        float sum1 = 0.0f;

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
                    const int weight_idx =
                        ((oc * kInputChannels + ic) * kKernelSize + kh) * kKernelSize + kw;
                    const float w = g_const_weights[weight_idx];
                    const int iw0 = ow0 * shape.stride + kw - shape.pad;
                    if (iw0 >= 0 && iw0 < shape.width) {
                        sum0 = fmaf(input[(ic * shape.height + ih) * shape.width + iw0], w, sum0);
                    }
                    if (ow1 < shape.out_w) {
                        const int iw1 = ow1 * shape.stride + kw - shape.pad;
                        if (iw1 >= 0 && iw1 < shape.width) {
                            sum1 = fmaf(input[(ic * shape.height + ih) * shape.width + iw1], w, sum1);
                        }
                    }
                }
            }
        }

        const size_t base = static_cast<size_t>(oc) * shape.out_h * shape.out_w +
                            static_cast<size_t>(oh) * shape.out_w;
        output[base + ow0] = sum0;
        if (ow1 < shape.out_w) {
            output[base + ow1] = sum1;
        }
    }
}

__global__ void cross_channel4_kernel(ConvShape shape,
                                      const float *__restrict__ input,
                                      float *__restrict__ output) {
    const size_t spatial = static_cast<size_t>(shape.out_h) * shape.out_w;
    const int groups = (shape.out_channels + 3) / 4;
    const size_t total = static_cast<size_t>(groups) * spatial;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += step) {
        const int group = static_cast<int>(idx / spatial);
        const int pos = static_cast<int>(idx % spatial);
        const int oh = pos / shape.out_w;
        const int ow = pos % shape.out_w;
        const int oc0 = group * 4;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

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
                    const float v = input[(ic * shape.height + ih) * shape.width + iw];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int oc = oc0 + j;
                        if (oc < shape.out_channels) {
                            const int weight_idx =
                                ((oc * kInputChannels + ic) * kKernelSize + kh) * kKernelSize + kw;
                            sums[j] = fmaf(v, g_const_weights[weight_idx], sums[j]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int oc = oc0 + j;
            if (oc < shape.out_channels) {
                output[static_cast<size_t>(oc) * spatial + pos] = sums[j];
            }
        }
    }
}

__global__ void implicit_im2col_kernel(ConvShape shape,
                                       const float *__restrict__ input,
                                       float *__restrict__ output) {
    __shared__ float weights[kChannelTile * kKernelElements];
    const int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;
    const int oc_base = blockIdx.y * kChannelTile;

    for (int i = local_tid; i < kChannelTile * kKernelElements; i += threads) {
        const int local_oc = i / kKernelElements;
        const int k = i % kKernelElements;
        const int oc = oc_base + local_oc;
        weights[i] = (oc < shape.out_channels) ? g_const_weights[oc * kKernelElements + k] : 0.0f;
    }
    __syncthreads();

    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_oc = threadIdx.y;
    const int oc = oc_base + local_oc;
    const int spatial = shape.out_h * shape.out_w;
    if (pos >= spatial || oc >= shape.out_channels) {
        return;
    }

    const int oh = pos / shape.out_w;
    const int ow = pos % shape.out_w;
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kKernelElements; ++k) {
        const int ic = k / (kKernelSize * kKernelSize);
        const int rem = k % (kKernelSize * kKernelSize);
        const int kh = rem / kKernelSize;
        const int kw = rem % kKernelSize;
        const int ih = oh * shape.stride + kh - shape.pad;
        const int iw = ow * shape.stride + kw - shape.pad;
        float v = 0.0f;
        if (ih >= 0 && ih < shape.height && iw >= 0 && iw < shape.width) {
            v = input[(ic * shape.height + ih) * shape.width + iw];
        }
        sum = fmaf(v, weights[local_oc * kKernelElements + k], sum);
    }

    output[static_cast<size_t>(oc) * spatial + pos] = sum;
}

__global__ void im2col_kernel(ConvShape shape,
                              const float *__restrict__ input,
                              float *__restrict__ columns) {
    const size_t n_cols = static_cast<size_t>(shape.out_h) * shape.out_w;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
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

void finish_result(MethodResult &result,
                   const ConvShape &shape,
                   const float *output,
                   const float *reference = nullptr) {
    result.gflops = shape.flops / (result.total.mean_ms * 1.0e6);
    result.checksum = checksum_device(output, shape.output_elems);
    if (reference != nullptr && reference != output) {
        result.max_abs_error = max_abs_diff_device(reference, output, shape.output_elems);
    }
}

MethodResult run_direct_baseline(const Options &opt,
                                 const ConvShape &shape,
                                 const float *input,
                                 float *reference_output) {
    MethodResult result;
    result.method = "direct_baseline";
    result.optimization = "baseline";
    result.notes = "one thread per output";
    const int blocks = launch_blocks(shape.output_elems);
    auto run_once = [&] {
        direct_conv_kernel<<<blocks, 256>>>(shape, input, reference_output);
    };
    result.total = benchmark(opt.warmup, opt.iters, run_once);
    finish_result(result, shape, reference_output);
    return result;
}

MethodResult run_shared_tile(const Options &opt,
                             const ConvShape &shape,
                             const float *input,
                             const float *reference) {
    MethodResult result;
    result.method = "shared_tile";
    result.optimization = "shared_memory_tiling";
    result.notes = "16x16 output tile, shared input tile";
    DeviceBuffer<float> output(shape.output_elems);
    const dim3 block(kTile, kTile);
    const dim3 grid(ceil_div(shape.out_w, kTile), ceil_div(shape.out_h, kTile), shape.out_channels);
    const int tile_h = (kTile - 1) * shape.stride + kKernelSize;
    const int tile_w = (kTile - 1) * shape.stride + kKernelSize;
    const size_t shared_bytes = static_cast<size_t>(kInputChannels) * tile_h * tile_w * sizeof(float);
    auto run_once = [&] {
        shared_tile_kernel<<<grid, block, shared_bytes>>>(shape, input, output.get());
    };
    result.total = benchmark(opt.warmup, opt.iters, run_once);
    finish_result(result, shape, output.get(), reference);
    return result;
}

MethodResult run_register_block2(const Options &opt,
                                 const ConvShape &shape,
                                 const float *input,
                                 const float *reference) {
    MethodResult result;
    result.method = "register_block2";
    result.optimization = "register_blocking";
    result.notes = "one thread computes two neighboring outputs";
    DeviceBuffer<float> output(shape.output_elems);
    const int pair_w = (shape.out_w + 1) / 2;
    const size_t pairs = static_cast<size_t>(shape.out_channels) * shape.out_h * pair_w;
    auto run_once = [&] {
        register_block2_kernel<<<launch_blocks(pairs), 256>>>(shape, input, output.get());
    };
    result.total = benchmark(opt.warmup, opt.iters, run_once);
    finish_result(result, shape, output.get(), reference);
    return result;
}

MethodResult run_cross_channel4(const Options &opt,
                                const ConvShape &shape,
                                const float *input,
                                const float *reference) {
    MethodResult result;
    result.method = "cross_channel4";
    result.optimization = "cross_output_channel_reuse";
    result.notes = "one thread reuses input patch for four filters";
    DeviceBuffer<float> output(shape.output_elems);
    const size_t spatial = static_cast<size_t>(shape.out_h) * shape.out_w;
    const size_t work = static_cast<size_t>((shape.out_channels + 3) / 4) * spatial;
    auto run_once = [&] {
        cross_channel4_kernel<<<launch_blocks(work), 256>>>(shape, input, output.get());
    };
    result.total = benchmark(opt.warmup, opt.iters, run_once);
    finish_result(result, shape, output.get(), reference);
    return result;
}

MethodResult run_implicit_im2col(const Options &opt,
                                 const ConvShape &shape,
                                 const float *input,
                                 const float *reference) {
    MethodResult result;
    result.method = "implicit_im2col";
    result.optimization = "implicit_im2col";
    result.notes = "GEMM-like block layout without materialized columns";
    DeviceBuffer<float> output(shape.output_elems);
    const int spatial = shape.out_h * shape.out_w;
    const dim3 block(16, kChannelTile);
    const dim3 grid(ceil_div(spatial, block.x), ceil_div(shape.out_channels, kChannelTile));
    auto run_once = [&] {
        implicit_im2col_kernel<<<grid, block>>>(shape, input, output.get());
    };
    result.total = benchmark(opt.warmup, opt.iters, run_once);
    finish_result(result, shape, output.get(), reference);
    return result;
}

MethodResult run_tensor_core_tf32(const Options &opt,
                                  const ConvShape &shape,
                                  const float *input,
                                  const float *weight,
                                  const float *reference,
                                  cublasHandle_t cublas) {
    MethodResult result;
    result.method = "tensor_core_tf32";
    result.optimization = "tensor_core_tf32_gemmex";
    result.notes = "explicit im2col plus cublasGemmEx FAST_TF32";

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
        CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n_cols, shape.out_channels,
                                  kKernelElements, &alpha, columns.get(), CUDA_R_32F, n_cols,
                                  weight, CUDA_R_32F, kKernelElements, &beta, output.get(),
                                  CUDA_R_32F, n_cols, CUBLAS_COMPUTE_32F_FAST_TF32,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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

void print_csv_header() {
    std::cout << "method,optimization,input_size,input_channels,kernel,out_channels,stride,padding,"
              << "out_h,out_w,warmup,iterations,mean_ms,std_ms,min_ms,gflops,workspace_mb,"
              << "im2col_ms,gemm_ms,max_abs_error,checksum,notes,gpu\n";
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
              << result.optimization << ','
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
              << csv_escape(result.notes) << ','
              << csv_escape(gpu) << '\n';
}

Options parse_options(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char *name) -> char * {
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
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opt.size <= 0 || opt.stride <= 0 || opt.out_channels <= 0 || opt.out_channels > 512) {
        throw std::runtime_error("invalid shape options");
    }
    return opt;
}

std::string current_gpu_name() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
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
        if (shape.weight_elems > kMaxConstWeights) {
            throw std::runtime_error("too many output channels for constant-memory weights");
        }

        std::vector<float> host_input(shape.input_elems);
        std::vector<float> host_weight(shape.weight_elems);
        fill_data(host_input, 17u + static_cast<uint32_t>(opt.size), 1.0f);
        fill_data(host_weight, 91u + static_cast<uint32_t>(opt.out_channels * 13 + opt.stride), 0.25f);

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
        std::vector<MethodResult> results;
        results.push_back(run_direct_baseline(opt, shape, input.get(), reference.get()));
        results.push_back(run_shared_tile(opt, shape, input.get(), reference.get()));
        results.push_back(run_register_block2(opt, shape, input.get(), reference.get()));
        results.push_back(run_cross_channel4(opt, shape, input.get(), reference.get()));
        results.push_back(run_implicit_im2col(opt, shape, input.get(), reference.get()));
        results.push_back(run_tensor_core_tf32(opt, shape, input.get(), weight.get(), reference.get(), cublas));

        for (const MethodResult &result : results) {
            print_result(result, opt, shape, gpu);
        }
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
