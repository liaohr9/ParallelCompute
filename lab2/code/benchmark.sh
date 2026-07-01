#!/bin/bash
# ============================================================
# 性能测试脚本：自动化运行不同进程数和矩阵规模的组合
# ============================================================
#
# 用途：
#   自动遍历所有进程数（1, 2, 4, 8, 16）和矩阵规模（128, 256, 512, 1024, 2048）
#   的组合，收集每次运行的时间开销，并输出为 CSV 格式表格。
#
# 使用方法：
#   chmod +x benchmark.sh
#   ./benchmark.sh

PROGRAM="./mpi_matrix"

# ---- 测试参数 ----
PROCESSES=(1 2 4 8 16)       # 进程数
SIZES=(128 256 512 1024)    # 矩阵规模（2048 时间较长，按需加入）

RUNS=3  # 每组参数重复运行次数，取平均值以减小波动

# ---- 编译程序 ----
echo "正在编译程序..."
make clean && make
if [ $? -ne 0 ]; then
    echo "编译失败！"
    exit 1
fi
echo "编译完成。"
echo ""

# ---- 检查最大进程数 ----
# macOS 默认 mpirun 最大进程数可能有限制，检查是否可用 16 进程
MAX_PROCS=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
echo "本机 CPU 核心数: $MAX_PROCS"

# ---- 输出 CSV 表头 ----
printf "%-10s" "Processes"
for size in "${SIZES[@]}"; do
    printf "%-15s" "${size}x${size}"
done
printf "\n"
printf "%-10s" "--------"
for size in "${SIZES[@]}"; do
    printf "%-15s" "--------"
done
printf "\n"

# ---- 运行测试并输出时间 ----
for np in "${PROCESSES[@]}"; do
    # 如果进程数超过逻辑核心数，跳过（避免严重超线程干扰）
    if [ "$np" -gt "$MAX_PROCS" ]; then
        printf "%-10s (跳过，超过核心数 %d)\n" "$np" "$MAX_PROCS"
        for size in "${SIZES[@]}"; do
            printf "%-15s" "N/A"
        done
        printf "\n"
        continue
    fi

    printf "%-10s" "$np"

    for size in "${SIZES[@]}"; do
        # 多次运行取平均
        total_time=0
        for run in $(seq 1 $RUNS); do
            # 从程序输出中提取计算时间（秒）
            time_output=$(mpirun -np "$np" "$PROGRAM" "$size" "$size" "$size" 2>/dev/null | grep "计算时间:" | awk '{print $3}')
            # 将秒转换为毫秒
            if [ -n "$time_output" ]; then
                ms=$(echo "$time_output * 1000" | bc -l 2>/dev/null || python3 -c "print($time_output * 1000)")
                total_time=$(echo "$total_time + $ms" | bc -l 2>/dev/null || python3 -c "print($total_time + $ms)")
            fi
        done

        # 计算平均值
        if [ $RUNS -gt 0 ]; then
            avg=$(echo "$total_time / $RUNS" | bc -l 2>/dev/null || python3 -c "print(f'{$total_time / $RUNS:.2f}')")
            printf "%-15s" "$avg"
        else
            printf "%-15s" "N/A"
        fi
    done
    printf "\n"
done

echo ""
echo "测试完成！"
echo "注：表中数值为 $RUNS 次运行的平均时间（毫秒）"
