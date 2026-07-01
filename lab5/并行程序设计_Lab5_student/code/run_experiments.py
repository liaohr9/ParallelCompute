#!/usr/bin/env python3
import csv
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results"


def run(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return next(csv.DictReader(lines))


def main():
    RESULTS.mkdir(exist_ok=True)
    subprocess.run(["make", "all"], cwd=ROOT, check=True)

    out = RESULTS / "experiment_results.csv"
    fieldnames = [
        "impl",
        "schedule",
        "M",
        "N",
        "K",
        "threads",
        "time_sec",
        "gflops",
        "checksum",
        "max_error",
    ]

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    def record(cmd):
        row = run(cmd)
        with out.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
        print(
            f"{row['impl']},{row['schedule']},n={row['M']},"
            f"threads={row['threads']},time={row['time_sec']}s",
            flush=True,
        )

    sizes = [512, 1024, 1536, 2048]
    threads = list(range(1, (os.cpu_count() or 8) + 1))

    record(["./openmp_gemm", "64", "64", "64", "4", "default"])
    record(["./pthread_gemm", "64", "64", "64", "4"])

    for size in sizes:
        for sched in ["default", "static1", "dynamic1"]:
            for th in threads:
                record(["./openmp_gemm", str(size), str(size), str(size), str(th), sched])

    for size in sizes:
        for th in threads:
            record(["./pthread_gemm", str(size), str(size), str(size), str(th)])
    print(out)


if __name__ == "__main__":
    main()
