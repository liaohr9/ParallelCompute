#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMG = ROOT / "报告" / "img"
IMG.mkdir(parents=True, exist_ok=True)


LABELS = {
    "naive_write_strided": "naive: coalesced read",
    "naive_read_strided": "naive: coalesced write",
    "tiled_nopad": "shared tile",
    "tiled_pad": "shared tile + padding",
}


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["n"] = int(row["n"])
            row["tile"] = int(row["tile"])
            row["block_rows"] = int(row["block_rows"])
            row["threads_per_block"] = int(row["threads_per_block"])
            row["repeat"] = int(row["repeat"])
            row["time_ms"] = float(row["time_ms"])
            row["bandwidth_gbs"] = float(row["bandwidth_gbs"])
            row["config"] = f'{row["tile"]}x{row["block_rows"]}'
            rows.append(row)
    return rows


def best_by_variant_and_size(rows):
    best = {}
    for row in rows:
        key = (row["variant"], row["n"])
        if key not in best or row["bandwidth_gbs"] > best[key]["bandwidth_gbs"]:
            best[key] = row
    return list(best.values())


def plot_best_assignment(rows):
    best = best_by_variant_and_size(rows)
    grouped = defaultdict(list)
    for row in sorted(best, key=lambda r: (r["variant"], r["n"])):
        grouped[row["variant"]].append(row)

    plt.figure(figsize=(8.4, 5.0))
    for variant, items in grouped.items():
        xs = [r["n"] for r in items]
        ys = [r["bandwidth_gbs"] for r in items]
        plt.plot(xs, ys, marker="o", linewidth=2, label=LABELS.get(variant, variant))
    plt.xlabel("Matrix size n")
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.title("Best bandwidth on assignment sizes")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG / "bandwidth_assignment_best.png", dpi=220)
    plt.close()


def plot_block_config(rows, n=2048):
    selected = [r for r in rows if r["n"] == n and r["variant"] == "tiled_pad"]
    selected.sort(key=lambda r: (r["tile"], r["block_rows"]))
    labels = [r["config"] for r in selected]
    ys = [r["bandwidth_gbs"] for r in selected]

    plt.figure(figsize=(8.4, 4.8))
    plt.bar(labels, ys, color="#2f7f7b")
    plt.xlabel("TILE_DIM x BLOCK_ROWS")
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.title(f"Block configuration impact, tiled_pad, n={n}")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(IMG / "block_config_2048.png", dpi=220)
    plt.close()


def plot_access_pattern(rows, n=2048):
    best = best_by_variant_and_size([r for r in rows if r["n"] == n])
    order = [
        "naive_write_strided",
        "naive_read_strided",
        "tiled_nopad",
        "tiled_pad",
    ]
    best_map = {r["variant"]: r for r in best}
    labels = [LABELS[v] for v in order if v in best_map]
    ys = [best_map[v]["bandwidth_gbs"] for v in order if v in best_map]

    plt.figure(figsize=(8.4, 4.8))
    plt.bar(labels, ys, color=["#8c6d31", "#636363", "#4f77aa", "#2f7f7b"])
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.title(f"Memory access pattern comparison, n={n}")
    plt.xticks(rotation=16, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(IMG / "access_pattern_2048.png", dpi=220)
    plt.close()


def plot_heavy(rows):
    best = best_by_variant_and_size(rows)
    grouped = defaultdict(list)
    for row in sorted(best, key=lambda r: (r["variant"], r["n"])):
        grouped[row["variant"]].append(row)

    plt.figure(figsize=(8.4, 5.0))
    for variant, items in grouped.items():
        xs = [r["n"] for r in items]
        ys = [r["bandwidth_gbs"] for r in items]
        plt.plot(xs, ys, marker="o", linewidth=2, label=LABELS.get(variant, variant))
    plt.xlabel("Matrix size n")
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.title("Scaling experiment on large matrices")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG / "bandwidth_heavy_best.png", dpi=220)
    plt.close()


def write_summary(assignment, heavy):
    best_assignment = best_by_variant_and_size(assignment)
    best_heavy = best_by_variant_and_size(heavy)

    overall = max(assignment + heavy, key=lambda r: r["bandwidth_gbs"])
    assignment_2048 = [r for r in best_assignment if r["n"] == 2048]
    heavy_best_pad = [
        r for r in best_heavy if r["variant"] == "tiled_pad"
    ]

    with open(RESULTS / "summary.md", "w") as f:
        f.write("# Benchmark summary\n\n")
        f.write(
            f"- Overall best: {overall['variant']} n={overall['n']} "
            f"config={overall['config']} bandwidth={overall['bandwidth_gbs']:.3f} GB/s "
            f"time={overall['time_ms']:.6f} ms\n"
        )
        f.write("- Best rows on n=2048:\n")
        for row in sorted(assignment_2048, key=lambda r: r["bandwidth_gbs"], reverse=True):
            f.write(
                f"  - {row['variant']} config={row['config']} "
                f"bandwidth={row['bandwidth_gbs']:.3f} GB/s "
                f"time={row['time_ms']:.6f} ms\n"
            )
        f.write("- Large-size tiled_pad best rows:\n")
        for row in sorted(heavy_best_pad, key=lambda r: r["n"]):
            f.write(
                f"  - n={row['n']} config={row['config']} "
                f"bandwidth={row['bandwidth_gbs']:.3f} GB/s "
                f"time={row['time_ms']:.6f} ms\n"
            )


def main():
    assignment = read_csv(RESULTS / "transpose_assignment.csv")
    heavy = read_csv(RESULTS / "transpose_heavy.csv")
    plot_best_assignment(assignment)
    plot_block_config(assignment)
    plot_access_pattern(assignment)
    plot_heavy(heavy)
    write_summary(assignment, heavy)


if __name__ == "__main__":
    main()
