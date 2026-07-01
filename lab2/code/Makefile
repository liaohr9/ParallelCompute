# ============================================================
# 编译：make
# 清理：make clean
# 测试：make test
# ============================================================

# 标准 MPI 编译器（Linux / 集群环境通用）
CXX = mpicxx
CXXFLAGS = -O3 -Wall

TARGETS = mpi_matrix mpi_cannon mpi_sparse

all: $(TARGETS)

mpi_matrix: mpi_matrix.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

mpi_cannon: mpi_cannon.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

mpi_sparse: mpi_sparse.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: $(TARGETS)
	@echo "--- 原始方案 (np=4, size=256) ---"
	mpirun -np 4 --oversubscribe ./mpi_matrix 256 256 256
	@echo ""
	@echo "--- Cannon 算法 (np=4, size=256) ---"
	mpirun -np 4 --oversubscribe ./mpi_cannon 256
	@echo ""
	@echo "--- 稀疏矩阵 (np=4, size=2048, sparsity=1%) ---"
	mpirun -np 4 --oversubscribe ./mpi_sparse 2048 0.01

clean:
	rm -f $(TARGETS)

.PHONY: all test clean
