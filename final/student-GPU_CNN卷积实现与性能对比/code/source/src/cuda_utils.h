#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <sstream>
#include <stdexcept>
#include <string>

inline std::string cuda_error(cudaError_t status, const char *expr, const char *file, int line) {
    std::ostringstream oss;
    oss << file << ":" << line << " CUDA error in " << expr << ": "
        << cudaGetErrorString(status);
    return oss.str();
}

inline std::string cublas_error(cublasStatus_t status, const char *expr, const char *file, int line) {
    std::ostringstream oss;
    oss << file << ":" << line << " cuBLAS error in " << expr << ": "
        << static_cast<int>(status);
    return oss.str();
}

inline std::string cudnn_error(cudnnStatus_t status, const char *expr, const char *file, int line) {
    std::ostringstream oss;
    oss << file << ":" << line << " cuDNN error in " << expr << ": "
        << cudnnGetErrorString(status);
    return oss.str();
}

#define CUDA_CHECK(expr)                                                                          \
    do {                                                                                          \
        cudaError_t _status = (expr);                                                             \
        if (_status != cudaSuccess) {                                                             \
            throw std::runtime_error(cuda_error(_status, #expr, __FILE__, __LINE__));             \
        }                                                                                         \
    } while (0)

#define CUBLAS_CHECK(expr)                                                                        \
    do {                                                                                          \
        cublasStatus_t _status = (expr);                                                          \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                                    \
            throw std::runtime_error(cublas_error(_status, #expr, __FILE__, __LINE__));           \
        }                                                                                         \
    } while (0)

#define CUDNN_CHECK(expr)                                                                         \
    do {                                                                                          \
        cudnnStatus_t _status = (expr);                                                           \
        if (_status != CUDNN_STATUS_SUCCESS) {                                                     \
            throw std::runtime_error(cudnn_error(_status, #expr, __FILE__, __LINE__));            \
        }                                                                                         \
    } while (0)

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { reset(count); }
    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() const { return ptr_; }
    size_t count() const { return count_; }

    void reset(size_t count) {
        if (ptr_ != nullptr) {
            CUDA_CHECK(cudaFree(ptr_));
        }
        ptr_ = nullptr;
        count_ = count;
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

  private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
};

class CudaEvents {
  public:
    CudaEvents() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaEvents() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

  private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

class CublasHandle {
  public:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
#if CUDART_VERSION >= 11000
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH));
#endif
    }
    ~CublasHandle() { cublasDestroy(handle_); }
    operator cublasHandle_t() const { return handle_; }

  private:
    cublasHandle_t handle_ = nullptr;
};

class CudnnHandle {
  public:
    CudnnHandle() { CUDNN_CHECK(cudnnCreate(&handle_)); }
    ~CudnnHandle() { cudnnDestroy(handle_); }
    operator cudnnHandle_t() const { return handle_; }

  private:
    cudnnHandle_t handle_ = nullptr;
};
