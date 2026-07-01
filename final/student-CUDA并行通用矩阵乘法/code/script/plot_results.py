#!/usr/bin/env python3
import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KERNEL_ORDER = ["global2d", "shared_tiled", "global1d", "global2d_btrans"]
SIZE_ORDER = [
    "512x512x512",
    "512x1024x1536",
    "1024x512x2048",
    "1024x1024x1024",
    "1536x1024x512",
    "1536x1536x1536",
    "2048x1536x1024",
    "2048x2048x2048",
]
BLOCK_ORDER_2D = ["8x8", "16x16", "32x8", "8x32", "32x16", "16x32", "32x32"]
BLOCK_ORDER_1D = ["128", "256", "512", "1024"]
COLORS = ["#2563a8", "#2f8f5b", "#d08a24", "#8b5fbf", "#b33f62"]


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    int_fields = ["m", "n", "k", "block_x", "block_y", "tile_k", "run_id"]
    float_fields = ["avg_ms", "min_ms", "max_ms", "gflops", "max_abs_error", "checksum"]
    for row in rows:
        for field in int_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
        row["size"] = f"{row['m']}x{row['n']}x{row['k']}"
        row["block"] = f"{row['block_x']}x{row['block_y']}"
        row["block_1d"] = str(row["block_x"])
    return rows


def grouped(rows, key):
    result = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return result


def stats(values):
    return {
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def ordered_labels(labels, preferred):
    seen = set(labels)
    ordered = [label for label in preferred if label in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)


def style_axes(ax):
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_overall_distribution(rows, output_dir):
    values = [row["gflops"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.hist(values, bins=18, color=COLORS[0], edgecolor="white", alpha=0.88)
    ax.axvline(statistics.mean(values), color=COLORS[2], linewidth=2, label="Mean")
    ax.axvline(statistics.median(values), color=COLORS[3], linewidth=2, label="Median")
    ax.set_title("Overall GFLOP/s Distribution")
    ax.set_xlabel("GFLOP/s")
    ax.set_ylabel("Cases")
    ax.legend(frameon=False)
    style_axes(ax)
    save(fig, output_dir, "overall_gflops_distribution.png")


def plot_kernel_summary(rows, output_dir):
    by_kernel = grouped(rows, lambda row: row["kernel"])
    kernels = ordered_labels(by_kernel.keys(), KERNEL_ORDER)
    metrics = ["avg", "median", "max"]
    width = 0.23
    x = list(range(len(kernels)))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, metric in enumerate(metrics):
        values = [stats([row["gflops"] for row in by_kernel[kernel]])[metric] for kernel in kernels]
        positions = [pos + (offset - 1) * width for pos in x]
        ax.bar(positions, values, width=width, label=metric.capitalize(), color=COLORS[offset])

    ax.set_title("Kernel Performance Summary")
    ax.set_xlabel("Kernel")
    ax.set_ylabel("GFLOP/s")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=12, ha="right")
    ax.legend(frameon=False, ncol=3)
    style_axes(ax)
    save(fig, output_dir, "kernel_summary.png")


def plot_best_by_kernel(rows, output_dir):
    by_kernel = grouped(rows, lambda row: row["kernel"])
    kernels = ordered_labels(by_kernel.keys(), KERNEL_ORDER)
    values = [max(row["gflops"] for row in by_kernel[kernel]) for kernel in kernels]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.barh(range(len(kernels)), values, color=COLORS[: len(kernels)], edgecolor="white")
    best_idx = max(range(len(values)), key=lambda idx: values[idx])
    bars[best_idx].set_edgecolor("#111111")
    bars[best_idx].set_linewidth(2.0)

    for idx, value in enumerate(values):
        ax.text(value + 70, idx, f"{value:.2f}", va="center", fontsize=10)

    ax.set_title("Best GFLOP/s by Kernel")
    ax.set_xlabel("Best GFLOP/s")
    ax.set_ylabel("Kernel")
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels(kernels)
    ax.set_xlim(0, max(values) * 1.15)
    style_axes(ax)
    save(fig, output_dir, "best_by_kernel.png")


def plot_best_by_size(rows, output_dir):
    best = {}
    for row in rows:
        current = best.get(row["size"])
        if current is None or row["gflops"] > current["gflops"]:
            best[row["size"]] = row

    labels = ordered_labels(best.keys(), SIZE_ORDER)
    values = [best[label]["gflops"] for label in labels]
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.bar(range(len(labels)), values, color=COLORS[1], edgecolor="white")
    ax.set_title("Best Configuration per Matrix Size")
    ax.set_xlabel("Matrix size")
    ax.set_ylabel("Best GFLOP/s")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    style_axes(ax)
    save(fig, output_dir, "best_by_size.png")


def plot_kernel_by_size(rows, output_dir):
    by_size_kernel = grouped(rows, lambda row: (row["size"], row["kernel"]))
    sizes = ordered_labels({row["size"] for row in rows}, SIZE_ORDER)
    kernels = ordered_labels({row["kernel"] for row in rows}, KERNEL_ORDER)
    width = 0.18
    x = list(range(len(sizes)))

    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    for idx, kernel in enumerate(kernels):
        values = []
        for size in sizes:
            group = by_size_kernel.get((size, kernel), [])
            values.append(statistics.mean(row["gflops"] for row in group) if group else 0.0)
        positions = [pos + (idx - (len(kernels) - 1) / 2) * width for pos in x]
        ax.bar(positions, values, width=width, label=kernel, color=COLORS[idx % len(COLORS)])

    ax.set_title("Average Kernel Performance by Matrix Size")
    ax.set_xlabel("Matrix size")
    ax.set_ylabel("Average GFLOP/s")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=30, ha="right")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    save(fig, output_dir, "kernel_by_size.png")


def plot_block_summary(rows, output_dir, kernel, filename, title, block_order):
    selected = [row for row in rows if row["kernel"] == kernel]
    block_key = (lambda row: row["block_1d"]) if kernel == "global1d" else (lambda row: row["block"])
    by_block = grouped(selected, block_key)
    blocks = ordered_labels(by_block.keys(), block_order)
    metrics = [("min", "Worst"), ("avg", "Average"), ("max", "Best")]
    width = 0.23
    x = list(range(len(blocks)))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for idx, (metric, label) in enumerate(metrics):
        values = [stats([row["gflops"] for row in by_block[block]])[metric] for block in blocks]
        positions = [pos + (idx - 1) * width for pos in x]
        ax.bar(positions, values, width=width, label=label, color=COLORS[idx])

    ax.set_title(title)
    ax.set_xlabel("Block")
    ax.set_ylabel("GFLOP/s")
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.legend(frameon=False, ncol=3)
    style_axes(ax)
    save(fig, output_dir, filename)


def plot_square_size_trend(rows, output_dir):
    best = {}
    for row in rows:
        if row["m"] != row["n"] or row["n"] != row["k"]:
            continue
        current = best.get(row["m"])
        if current is None or row["gflops"] > current["gflops"]:
            best[row["m"]] = row

    xs = sorted(best)
    ys = [best[x]["gflops"] for x in xs]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(xs, ys, marker="o", linewidth=2.2, color=COLORS[0])
    ax.set_title("Best GFLOP/s for Square Matrices")
    ax.set_xlabel("N for NxNxN")
    ax.set_ylabel("Best GFLOP/s")
    ax.set_xticks(xs)
    style_axes(ax)
    save(fig, output_dir, "square_size_trend.png")


def plot_task_split(rows, output_dir):
    kernels = ["global2d", "global1d"]
    by_kernel = grouped(rows, lambda row: row["kernel"])
    metrics = [("avg", "Average"), ("max", "Best")]
    width = 0.32
    x = list(range(len(kernels)))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for idx, (metric, label) in enumerate(metrics):
        values = [stats([row["gflops"] for row in by_kernel[kernel]])[metric] for kernel in kernels]
        positions = [pos + (idx - 0.5) * width for pos in x]
        ax.bar(positions, values, width=width, label=label, color=COLORS[idx])

    ax.set_title("2D vs 1D Task Mapping")
    ax.set_xlabel("Kernel")
    ax.set_ylabel("GFLOP/s")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels)
    ax.legend(frameon=False)
    style_axes(ax)
    save(fig, output_dir, "task_split.png")


def plot_correctness(rows, output_dir):
    by_kernel = grouped(rows, lambda row: row["kernel"])
    kernels = ordered_labels(by_kernel.keys(), KERNEL_ORDER)
    metrics = [("avg", "Mean max_abs_error"), ("max", "Max max_abs_error")]
    width = 0.32
    x = list(range(len(kernels)))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for idx, (metric, label) in enumerate(metrics):
        values = [
            stats([row["max_abs_error"] for row in by_kernel[kernel]])[metric]
            for kernel in kernels
        ]
        positions = [pos + (idx - 0.5) * width for pos in x]
        ax.bar(positions, values, width=width, label=label, color=COLORS[idx])

    ax.set_title("Correctness Error by Kernel")
    ax.set_xlabel("Kernel")
    ax.set_ylabel("Absolute error")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=12, ha="right")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False)
    style_axes(ax)
    save(fig, output_dir, "correctness_errors.png")


def main():
    parser = argparse.ArgumentParser(description="Generate matplotlib figures for the CUDA report.")
    parser.add_argument(
        "--csv",
        default="results/matrix_mul_results_20260615_003634.csv",
        help="Input CSV generated by run_experiments.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="报告/img",
        help="Directory for generated PNG figures.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    plot_overall_distribution(rows, args.output_dir)
    plot_kernel_summary(rows, args.output_dir)
    plot_best_by_kernel(rows, args.output_dir)
    plot_best_by_size(rows, args.output_dir)
    plot_kernel_by_size(rows, args.output_dir)
    plot_block_summary(
        rows,
        args.output_dir,
        "global2d",
        "block_global2d.png",
        "global2d Performance by Block Shape",
        BLOCK_ORDER_2D,
    )
    plot_block_summary(
        rows,
        args.output_dir,
        "shared_tiled",
        "block_shared_tiled.png",
        "shared_tiled Performance by Block Shape",
        BLOCK_ORDER_2D,
    )
    plot_block_summary(
        rows,
        args.output_dir,
        "global2d_btrans",
        "block_global2d_btrans.png",
        "global2d_btrans Performance by Block Shape",
        BLOCK_ORDER_2D,
    )
    plot_block_summary(
        rows,
        args.output_dir,
        "global1d",
        "block_global1d.png",
        "global1d Performance by Block Size",
        BLOCK_ORDER_1D,
    )
    plot_square_size_trend(rows, args.output_dir)
    plot_task_split(rows, args.output_dir)
    plot_correctness(rows, args.output_dir)


if __name__ == "__main__":
    main()
