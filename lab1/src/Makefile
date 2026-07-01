MKLROOT ?= /opt/intel/oneapi/mkl/2025.3
CC := gcc

# 基础编译选项：统一启用 C11、警告和头文件目录。
CFLAGS_BASE := -std=c11 -Wall -Wextra -I./src
# O0 对应未做编译优化的基线版本。
CFLAGS_O0 := $(CFLAGS_BASE) -O0
# O3 + march=native 对应“编译优化”版本，也供展开版和 MKL 版复用。
CFLAGS_O3 := $(CFLAGS_BASE) -O3 -march=native
LDFLAGS_COMMON := -lm

# MKL 的头文件与链接参数。
# 这里显式选择 sequential 版本，保证与前面几个串行版本对比时口径一致。
MKL_INC := -I$(MKLROOT)/include
MKL_LIB := -L$(MKLROOT)/lib -Wl,-rpath,$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

BIN_DIR := bin
SRC_DIR := src

COMMON_SRC := $(SRC_DIR)/common.c
KERNELS_SRC := $(SRC_DIR)/kernels.c

all: $(BIN_DIR)/bench_o0 $(BIN_DIR)/bench_loop $(BIN_DIR)/bench_opt $(BIN_DIR)/bench_unroll $(BIN_DIR)/mkl_bench

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# 2 C/C++：朴素 ijk 版本，不做编译优化。
$(BIN_DIR)/bench_o0: $(COMMON_SRC) $(KERNELS_SRC) $(SRC_DIR)/bench.c | $(BIN_DIR)
	$(CC) $(CFLAGS_O0) -o $@ $(COMMON_SRC) $(KERNELS_SRC) $(SRC_DIR)/bench.c $(LDFLAGS_COMMON)

# 3 调整循环顺序：仍然 O0，只更换 kernel，确保变量尽量单一。
$(BIN_DIR)/bench_loop: $(COMMON_SRC) $(KERNELS_SRC) $(SRC_DIR)/bench.c | $(BIN_DIR)
	$(CC) $(CFLAGS_O0) -o $@ $(COMMON_SRC) $(KERNELS_SRC) $(SRC_DIR)/bench.c $(LDFLAGS_COMMON)

# 4 编译优化：切到 O3 与本机架构优化。
$(BIN_DIR)/bench_opt: $(COMMON_SRC) $(KERNELS_SRC) $(SRC_DIR)/bench.c | $(BIN_DIR)
	$(CC) $(CFLAGS_O3) -o $@ $(COMMON_SRC) $(KERNELS_SRC) $(SRC_DIR)/bench.c $(LDFLAGS_COMMON)

# 5 循环展开：沿用 O3，只更换为 unroll4 kernel。
$(BIN_DIR)/bench_unroll: $(COMMON_SRC) $(KERNELS_SRC) $(SRC_DIR)/bench.c | $(BIN_DIR)
	$(CC) $(CFLAGS_O3) -o $@ $(COMMON_SRC) $(KERNELS_SRC) $(SRC_DIR)/bench.c $(LDFLAGS_COMMON)

# 6 Intel MKL：独立驱动程序，直接调用 cblas_dgemm。
$(BIN_DIR)/mkl_bench: $(COMMON_SRC) $(SRC_DIR)/mkl_bench.c | $(BIN_DIR)
	$(CC) $(CFLAGS_O3) $(MKL_INC) -o $@ $(COMMON_SRC) $(SRC_DIR)/mkl_bench.c $(MKL_LIB)

# 先编译，再执行快速正确性校验。
check: all
	python3 scripts/run_experiments.py --check-only

# 执行完整实验并刷新 results 目录中的汇总文件。
bench: all
	python3 scripts/run_experiments.py

clean:
	rm -rf $(BIN_DIR) results/*.csv results/*.md

.PHONY: all check bench clean
