#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << ": " << cudaGetErrorString(err__) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__ void hello_kernel() {
    printf("Hello World from Thread (%d, %d) in Block %d!\n",
           threadIdx.x, threadIdx.y, blockIdx.x);
}

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog << " n m k\n"
              << "  n: number of blocks, range [1, 32]\n"
              << "  m: blockDim.x, range [1, 32]\n"
              << "  k: blockDim.y, range [1, 32]\n";
}

int main(int argc, char **argv) {
    int n = 2;
    int m = 3;
    int k = 4;

    if (argc == 4) {
        n = std::atoi(argv[1]);
        m = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    } else if (argc != 1) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (n < 1 || n > 32 || m < 1 || m > 32 || k < 1 || k > 32) {
        std::cerr << "Error: n, m and k must all be in [1, 32].\n";
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::cout << "Launching " << n << " blocks, each block has "
              << m << " x " << k << " threads." << std::endl;

    dim3 grid(n);
    dim3 block(m, k);
    hello_kernel<<<grid, block>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Hello World from the host!" << std::endl;
    CHECK_CUDA(cudaDeviceReset());
    return EXIT_SUCCESS;
}
