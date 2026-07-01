CUDA_HOME ?= /usr/local/cuda
CUDNN_HOME ?= /home/haoran/miniconda3/envs/act/lib/python3.10/site-packages/nvidia/cudnn
CUDA_ARCH ?= sm_120
NVCC ?= nvcc

BIN_DIR := bin
SRC_DIR := src
TARGET := $(BIN_DIR)/conv_benchmark

NVCCFLAGS ?= -O3 -std=c++17 --use_fast_math -lineinfo -arch=$(CUDA_ARCH)
INCLUDES := -I$(CUDA_HOME)/include -I$(CUDNN_HOME)/include
LIB_DIRS := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/targets/x86_64-linux/lib -L$(CUDNN_HOME)/lib
RPATHS := -Xlinker -rpath -Xlinker $(CUDNN_HOME)/lib -Xlinker -rpath -Xlinker $(CUDA_HOME)/targets/x86_64-linux/lib
LIBS := -lcublas -l:libcudnn.so.9 -l:libcudnn_cnn.so.9 -l:libcudnn_ops.so.9 -l:libcudnn_graph.so.9 -l:libcudnn_heuristic.so.9

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC_DIR)/conv_benchmark.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LIB_DIRS) $(RPATHS) $(LIBS)

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BIN_DIR)
