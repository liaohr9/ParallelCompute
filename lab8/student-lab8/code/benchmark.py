#!/usr/bin/env python3
"""Build, run, and plot the OpenMP shortest-path experiment."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
IMG = ROOT / "报告" / "img"
BIN = BUILD / "shortest_path_omp"

DATASETS = {
    "flower": ROOT / "data" / "updated_flower.csv",
    "mouse": ROOT / "data" / "updated_mouse.csv",
}


def run(cmd: list[str], env: dict[str, str] | None = None) -> str:
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)
    return completed.stdout


def build() -> None:
    BUILD.mkdir(exist_ok=True)
    run(
        [
            "g++",
            "-std=c++17",
            "-O3",
            "-march=native",
            "-flto",
            "-fopenmp",
            "-DNDEBUG",
            str(ROOT / "code" / "shortest_path_omp.cpp"),
            "-o",
            str(BIN),
        ]
    )


def generate_queries() -> dict[str, Path]:
    query_paths: dict[str, Path] = {}
    query_dir = RESULTS / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    for name, graph in DATASETS.items():
        output = query_dir / f"{name}_all_pairs.csv"
        run(
            [
                "python3",
                str(ROOT / "code" / "generate_queries.py"),
                str(graph),
                str(output),
                "--mode",
                "all-pairs",
            ]
        )
        query_paths[name] = output
    return query_paths


def parse_stats(line: str) -> dict[str, str]:
    stats: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z_]+)=([^ ]+)", line):
        stats[key] = value
    return stats


def benchmark(query_paths: dict[str, Path], threads: list[int], repeat: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    env_base = os.environ.copy()
    env_base["OMP_PROC_BIND"] = "spread"
    env_base["OMP_PLACES"] = "cores"

    for dataset, graph in DATASETS.items():
        for t in threads:
            env = env_base.copy()
            env["OMP_NUM_THREADS"] = str(t)
            output = run(
                [
                    str(BIN),
                    str(graph),
                    str(query_paths[dataset]),
                    "-",
                    str(repeat),
                    "--no-output",
                ],
                env=env,
            )
            stats = parse_stats(output.strip().splitlines()[-1])
            stats["dataset"] = dataset
            rows.append(stats)
    return rows


def write_benchmark_csv(rows: list[dict[str, str]]) -> Path:
    path = RESULTS / "benchmark_results.csv"
    fields = [
        "dataset",
        "threads",
        "vertices",
        "csv_edges",
        "undirected_arcs",
        "queries",
        "active_sources",
        "repeat",
        "compute_seconds_total",
        "compute_seconds_avg",
        "checksum",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def write_outputs(query_paths: dict[str, Path], threads: int) -> None:
    output_dir = RESULTS / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "cores"
    env["OMP_NUM_THREADS"] = str(threads)
    for dataset, graph in DATASETS.items():
        run(
            [
                str(BIN),
                str(graph),
                str(query_paths[dataset]),
                str(output_dir / f"{dataset}_shortest_paths.csv"),
                "1",
            ],
            env=env,
        )


def plot(rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMG.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=180)
    for dataset, dataset_rows in sorted(by_dataset.items()):
        dataset_rows.sort(key=lambda r: int(r["threads"]))
        threads = [int(r["threads"]) for r in dataset_rows]
        times = [float(r["compute_seconds_avg"]) for r in dataset_rows]
        base = times[0]
        speedups = [base / value for value in times]
        efficiency = [speedup / thread for speedup, thread in zip(speedups, threads)]

        axes[0].plot(threads, times, marker="o", linewidth=1.8, label=dataset)
        axes[1].plot(threads, speedups, marker="o", linewidth=1.8, label=dataset)

        table_path = RESULTS / f"{dataset}_speedup.csv"
        with table_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["threads", "time_s", "speedup", "efficiency"])
            for thread, time_s, speedup, eff in zip(threads, times, speedups, efficiency):
                writer.writerow([thread, f"{time_s:.6f}", f"{speedup:.4f}", f"{eff:.4f}"])

    axes[0].set_xlabel("OpenMP threads")
    axes[0].set_ylabel("Average compute time (s)")
    axes[0].set_title("Runtime")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_xlabel("OpenMP threads")
    axes[1].set_ylabel("Speedup vs 1 thread")
    axes[1].set_title("Speedup")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    for ax in axes:
        ax.set_xticks(sorted({int(r["threads"]) for r in rows}))

    fig.tight_layout()
    fig.savefig(IMG / "benchmark_results.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", default="1,2,4,8,16,32")
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-outputs", action="store_true")
    args = parser.parse_args()

    threads = [int(x) for x in args.threads.split(",") if x.strip()]
    RESULTS.mkdir(exist_ok=True)
    if not args.skip_build:
        build()
    query_paths = generate_queries()
    rows = benchmark(query_paths, threads, args.repeat)
    csv_path = write_benchmark_csv(rows)
    print(f"benchmark_csv={csv_path}", flush=True)
    if not args.skip_outputs:
        write_outputs(query_paths, threads=max(t for t in threads if t <= 16))
    plot(rows)
    print(f"plot={IMG / 'benchmark_results.png'}", flush=True)


if __name__ == "__main__":
    main()
