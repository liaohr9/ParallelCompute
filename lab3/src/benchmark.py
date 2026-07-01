#!/usr/bin/env python3
"""
Benchmark script for MPI parallel matrix multiplication implementations.

Runs all implementations with different process counts and matrix sizes,
collects timing data, and generates analysis tables and plots.
"""

import subprocess
import os
import re
import json
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# === Configuration ===

IMPLEMENTATIONS = {
    'collective': {
        'binary': './matmul_collective',
        'label': 'MPI Collective (Scatter/Gather)',
    },
    'struct': {
        'binary': './matmul_struct',
        'label': 'MPI Struct Datatype (Multiple Structs)',
    },
    'p2p': {
        'binary': './matmul_p2p',
        'label': 'MPI Point-to-Point (Send/Recv)',
    },
    'column_block': {
        'binary': './matmul_column_block',
        'label': 'Column-Block Partitioning (Allreduce)',
    },
}

# Matrix sizes and process counts from experiment requirements
MATRIX_SIZES = [128, 256, 512, 1024, 2048]
PROC_COUNTS = [1, 2, 4, 8, 16]

# Number of warm-up runs and actual runs for averaging
WARMUP_RUNS = 1
ACTUAL_RUNS = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'benchmark_results.json')


def run_single(binary, m, n, k, num_procs):
    """Run one MPI program and return elapsed time in seconds, or None on failure."""
    cmd = ['mpirun', '--allow-run-as-root', '-np', str(num_procs), binary,
           str(m), str(n), str(k)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per run
            cwd=SCRIPT_DIR
        )
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr.strip()[-200:]}")
            return None

        # Parse time from stdout
        match = re.search(r'Time \(compute only\):\s+([\d.]+)\s+seconds', result.stdout)
        if match:
            return float(match.group(1))
        else:
            print(f"  Could not parse time from output: {result.stdout[-200:]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def benchmark_single(impl_name, m, n, k, num_procs):
    """Run multiple times and return average time."""
    binary = IMPLEMENTATIONS[impl_name]['binary']
    times = []

    # Warm-up
    for _ in range(WARMUP_RUNS):
        run_single(binary, m, n, k, num_procs)

    # Actual runs
    for i in range(ACTUAL_RUNS):
        t = run_single(binary, m, n, k, num_procs)
        if t is not None:
            times.append(t)
        else:
            print(f"  Run {i+1} failed, skipping")

    if times:
        avg_time = sum(times) / len(times)
        return avg_time
    return None


def run_benchmarks():
    """Run full benchmark suite."""
    results = {}

    total_tasks = len(IMPLEMENTATIONS) * len(MATRIX_SIZES) * len(PROC_COUNTS)
    task_num = 0

    for impl_name, impl_info in IMPLEMENTATIONS.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {impl_info['label']}")
        print(f"{'='*60}")
        results[impl_name] = {}

        for size in MATRIX_SIZES:
            results[impl_name][str(size)] = {}
            print(f"\n  Matrix size: {size}x{size}")

            for np_val in PROC_COUNTS:
                task_num += 1
                # Skip if not divisible
                if size % np_val != 0:
                    results[impl_name][str(size)][str(np_val)] = None
                    print(f"    np={np_val:2d}: SKIPPED (size not divisible)")
                    continue

                print(f"    np={np_val:2d}: running...", end='', flush=True)
                avg_t = benchmark_single(impl_name, size, size, size, np_val)
                results[impl_name][str(size)][str(np_val)] = avg_t
                if avg_t is not None:
                    print(f" {avg_t:.6f}s")
                else:
                    print(f" FAILED")

                # Save intermediate results
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=2)

        print(f"\n  Done: {impl_info['label']}")

    return results


def print_table(results, impl_name):
    """Print a formatted table for one implementation."""
    label = IMPLEMENTATIONS[impl_name]['label']
    print(f"\n{'='*70}")
    print(f"Table: {label}")
    print(f"{'='*70}")

    # Header
    header = f"{'进程数':>6}"
    for size in MATRIX_SIZES:
        header += f" | {size:>8}"
    print(header)
    print("-" * len(header))

    for np_val in PROC_COUNTS:
        row = f"{np_val:>6}"
        for size in MATRIX_SIZES:
            t = results.get(impl_name, {}).get(str(size), {}).get(str(np_val))
            if t is not None:
                row += f" | {t:>8.4f}"
            else:
                row += f" | {'N/A':>8}"
        print(row)


def generate_plots(results, output_dir=None):
    """Generate performance analysis plots."""
    if output_dir is None:
        output_dir = SCRIPT_DIR

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    # Plot 1: Speedup vs Process Count (for each matrix size)
    fig, axes = plt.subplots(1, len(MATRIX_SIZES), figsize=(20, 4), dpi=150)
    if len(MATRIX_SIZES) == 1:
        axes = [axes]

    for idx, size in enumerate(MATRIX_SIZES):
        ax = axes[idx]
        for i, (impl_name, impl_info) in enumerate(IMPLEMENTATIONS.items()):
            np_vals = []
            speedups = []
            for np_val in PROC_COUNTS:
                t = results.get(impl_name, {}).get(str(size), {}).get(str(np_val))
                t1 = results.get(impl_name, {}).get(str(size), {}).get('1')
                if t is not None and t1 is not None and t > 0 and np_val > 1:
                    np_vals.append(np_val)
                    speedups.append(t1 / t)
            if np_vals:
                ax.plot(np_vals, speedups, marker=markers[i], color=colors[i],
                       label=impl_info['label'], linewidth=1.5, markersize=6)

        ax.plot(PROC_COUNTS, PROC_COUNTS, 'k--', alpha=0.3, label='Ideal')
        ax.set_xlabel('Process Count', fontsize=10)
        ax.set_title(f'Speedup (size={size})', fontsize=11)
        ax.set_xticks(PROC_COUNTS)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Speedup', fontsize=10)

    axes[0].legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_speedup.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: plot_speedup.png")

    # Plot 2: Execution Time vs Matrix Size (for each process count)
    fig, axes = plt.subplots(1, len(PROC_COUNTS), figsize=(20, 4), dpi=150)
    if len(PROC_COUNTS) == 1:
        axes = [axes]

    for idx, np_val in enumerate(PROC_COUNTS):
        ax = axes[idx]
        for i, (impl_name, impl_info) in enumerate(IMPLEMENTATIONS.items()):
            sizes = []
            times = []
            for size in MATRIX_SIZES:
                t = results.get(impl_name, {}).get(str(size), {}).get(str(np_val))
                if t is not None:
                    sizes.append(size)
                    times.append(t)
            if sizes:
                ax.plot(sizes, times, marker=markers[i], color=colors[i],
                       label=impl_info['label'], linewidth=1.5, markersize=6)

        ax.set_xlabel('Matrix Size', fontsize=10)
        ax.set_title(f'Time (np={np_val})', fontsize=11)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(MATRIX_SIZES)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Time (s)', fontsize=10)

    axes[0].legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_time.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: plot_time.png")

    # Plot 3: Parallel Efficiency (speedup / num_procs * 100%)
    fig, axes = plt.subplots(1, len(MATRIX_SIZES), figsize=(20, 4), dpi=150)
    if len(MATRIX_SIZES) == 1:
        axes = [axes]

    for idx, size in enumerate(MATRIX_SIZES):
        ax = axes[idx]
        for i, (impl_name, impl_info) in enumerate(IMPLEMENTATIONS.items()):
            np_vals = []
            efficiency = []
            for np_val in PROC_COUNTS:
                t = results.get(impl_name, {}).get(str(size), {}).get(str(np_val))
                t1 = results.get(impl_name, {}).get(str(size), {}).get('1')
                if t is not None and t1 is not None and t > 0 and np_val > 1:
                    np_vals.append(np_val)
                    efficiency.append((t1 / t) / np_val * 100.0)
            if np_vals:
                ax.plot(np_vals, efficiency, marker=markers[i], color=colors[i],
                       label=impl_info['label'], linewidth=1.5, markersize=6)

        ax.axhline(y=100, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Process Count', fontsize=10)
        ax.set_title(f'Parallel Efficiency (size={size})', fontsize=11)
        ax.set_xticks(PROC_COUNTS)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Efficiency (%)', fontsize=10)

    axes[0].legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_efficiency.png'), bbox_inches='tight')
    plt.close()
    print("  Saved: plot_efficiency.png")

    # Plot 4: Heatmap of execution times
    for impl_name, impl_info in IMPLEMENTATIONS.items():
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        data = np.zeros((len(PROC_COUNTS), len(MATRIX_SIZES)))
        for i, np_val in enumerate(PROC_COUNTS):
            for j, size in enumerate(MATRIX_SIZES):
                t = results.get(impl_name, {}).get(str(size), {}).get(str(np_val))
                if t is not None:
                    data[i, j] = t
                else:
                    data[i, j] = np.nan

        im = ax.imshow(data, aspect='auto', cmap='YlOrRd', origin='upper')
        ax.set_xticks(range(len(MATRIX_SIZES)))
        ax.set_xticklabels([str(s) for s in MATRIX_SIZES])
        ax.set_yticks(range(len(PROC_COUNTS)))
        ax.set_yticklabels([str(p) for p in PROC_COUNTS])
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Process Count')
        ax.set_title(f'Execution Time (s) - {impl_info["label"]}')

        # Add text annotations
        for i in range(len(PROC_COUNTS)):
            for j in range(len(MATRIX_SIZES)):
                if not np.isnan(data[i, j]) and data[i, j] > 0:
                    ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                           fontsize=7, color='black' if data[i, j] < data.max() * 0.5 else 'white')

        plt.colorbar(im, ax=ax, label='Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{impl_name}.png'), bbox_inches='tight')
        plt.close()
        print(f"  Saved: heatmap_{impl_name}.png")


def generate_analysis(results):
    """Generate textual performance analysis."""
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)

    for impl_name, impl_info in IMPLEMENTATIONS.items():
        print(f"\n--- {impl_info['label']} ---")

        # Speedup and efficiency analysis
        for size in MATRIX_SIZES:
            t1 = results.get(impl_name, {}).get(str(size), {}).get('1')
            t16 = results.get(impl_name, {}).get(str(size), {}).get('16')

            if t1 is not None and t16 is not None and t1 > 0 and t16 > 0:
                speedup = t1 / t16
                efficiency = speedup / 16 * 100
                print(f"  Size {size}: 1-proc={t1:.4f}s, 16-proc={t16:.4f}s, "
                      f"speedup={speedup:.2f}x, efficiency={efficiency:.1f}%")
            elif t1 is not None:
                print(f"  Size {size}: 1-proc={t1:.4f}s, 16-proc=N/A")

    # Comparison across implementations
    print("\n--- Implementation Comparison ---")
    print("For each matrix size and process count, best time highlighted:")
    for size in MATRIX_SIZES:
        for np_val in PROC_COUNTS:
            best_impl = None
            best_time = float('inf')
            times = {}
            for impl_name in IMPLEMENTATIONS:
                t = results.get(impl_name, {}).get(str(size), {}).get(str(np_val))
                if t is not None:
                    times[impl_name] = t
                    if t < best_time:
                        best_time = t
                        best_impl = impl_name

            if times:
                print(f"  size={size}, np={np_val}: best={best_impl} ({best_time:.4f}s)")


def main():
    print("="*60)
    print("MPI Matrix Multiplication Benchmark")
    print("="*60)

    # Check that binaries exist
    for impl_name, impl_info in IMPLEMENTATIONS.items():
        binary_path = os.path.join(SCRIPT_DIR, impl_info['binary'])
        if not os.path.isfile(binary_path):
            print(f"ERROR: Binary {binary_path} not found. Compile first!")
            sys.exit(1)

    # Run benchmarks
    results = run_benchmarks()

    # Print tables
    for impl_name in IMPLEMENTATIONS:
        print_table(results, impl_name)

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(results)

    # Generate analysis
    generate_analysis(results)

    # Save final results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")

    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
