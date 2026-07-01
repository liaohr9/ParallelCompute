NVCC ?= nvcc
TARGET ?= matrix_mul
CUDA_ARCH ?= sm_120

NVCCFLAGS ?= -O3 -std=c++17 --use_fast_math -Xptxas=-O3

.PHONY: all clean

all: $(TARGET)

$(TARGET): matrix_mul.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) $< -o $@

clean:
	rm -f $(TARGET)
