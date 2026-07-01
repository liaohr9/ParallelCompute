#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SUMMARY = PROJECT / "results" / "heated_plate_summary.csv"
OUT_DIR = Path("/tmp/lab6_plots")


def savefig(fig, name):
    fig.savefig(OUT_DIR / name, dpi=180)


def load_rows():
    with SUMMARY.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["threads"] = int(row["threads"])
        row["best_sec"] = float(row["best_sec"])
        row["median_sec"] = float(row["median_sec"])
        row["speedup_vs_openmp_1t"] = float(row["speedup_vs_openmp_1t"])
    return rows


def grouped(rows, impl, metric):
    schedules = ["static", "dynamic", "guided"]
    data = {}
    for schedule in schedules:
        subset = [r for r in rows if r["impl"] == impl and r["schedule"] == schedule]
        subset.sort(key=lambda r: r["threads"])
        data[schedule] = ([r["threads"] for r in subset], [r[metric] for r in subset])
    return data


def draw_time(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, impl, title in [
        (axes[0], "openmp", "OpenMP"),
        (axes[1], "pthread", "Pthreads"),
    ]:
        for schedule, (xs, ys) in grouped(rows, impl, "best_sec").items():
            ax.plot(xs, ys, marker="o", linewidth=2, label=schedule)
        ax.set_title(f"{title}: best time")
        ax.set_xlabel("Threads")
        ax.set_xticks(sorted({r["threads"] for r in rows}))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
    axes[0].set_ylabel("Time (s)")
    fig.tight_layout()
    savefig(fig, "heated_plate_time.png")
    plt.close(fig)


def draw_speedup(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, impl, title in [
        (axes[0], "openmp", "OpenMP"),
        (axes[1], "pthread", "Pthreads"),
    ]:
        for schedule, (xs, ys) in grouped(rows, impl, "speedup_vs_openmp_1t").items():
            ax.plot(xs, ys, marker="o", linewidth=2, label=schedule)
        ax.set_title(f"{title}: speedup")
        ax.set_xlabel("Threads")
        ax.set_xticks(sorted({r["threads"] for r in rows}))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
    axes[0].set_ylabel("Speedup vs OpenMP static 1 thread")
    fig.tight_layout()
    savefig(fig, "heated_plate_speedup.png")
    plt.close(fig)


def draw_best_compare(rows):
    labels = []
    values = []
    colors = []
    for impl in ["openmp", "pthread"]:
        for schedule in ["static", "dynamic", "guided"]:
            subset = [r for r in rows if r["impl"] == impl and r["schedule"] == schedule]
            best = min(subset, key=lambda r: r["best_sec"])
            labels.append(f"{impl}\n{schedule}\n{best['threads']}T")
            values.append(best["best_sec"])
            colors.append("#4c78a8" if impl == "openmp" else "#f58518")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Best time (s)")
    ax.set_title("Best configuration by implementation and schedule")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    savefig(fig, "heated_plate_best_compare.png")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    draw_time(rows)
    draw_speedup(rows)
    draw_best_compare(rows)
    print(OUT_DIR / "heated_plate_time.png")
    print(OUT_DIR / "heated_plate_speedup.png")
    print(OUT_DIR / "heated_plate_best_compare.png")


if __name__ == "__main__":
    main()
