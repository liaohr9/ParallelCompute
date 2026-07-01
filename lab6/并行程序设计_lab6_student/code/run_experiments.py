#!/usr/bin/env python3
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results"
RUN_TIMEOUT = os.environ.get("RUN_TIMEOUT")
RUN_TIMEOUT = int(RUN_TIMEOUT) if RUN_TIMEOUT else None


def run_once(cmd, progress, total):
    start = time.monotonic()
    timeout_msg = "no timeout" if RUN_TIMEOUT is None else f"timeout={RUN_TIMEOUT}s"
    print(f"[{progress}/{total}] start: {' '.join(cmd)} ({timeout_msg})", flush=True)
    kwargs = {
        "cwd": ROOT,
        "check": True,
        "text": True,
        "capture_output": True,
    }
    if RUN_TIMEOUT is not None:
        kwargs["timeout"] = RUN_TIMEOUT
    proc = subprocess.run(cmd, **kwargs)
    rows = [line for line in proc.stdout.splitlines() if line.strip()]
    row = next(csv.DictReader(rows))
    if row["iterations"] != "16966":
        raise RuntimeError(f"bad iteration count from {cmd}: {row}")
    if row["checksum"] != "21490.633363":
        raise RuntimeError(f"bad checksum from {cmd}: {row}")
    elapsed = time.monotonic() - start
    print(
        f"[{progress}/{total}] done: {row['impl']},{row['schedule']},"
        f"threads={row['threads']}, repeat={row.get('repeat', '0')}, "
        f"program_time={row['time_sec']}s, wall={elapsed:.2f}s, remaining={total - progress}",
        flush=True,
    )
    return row


def main():
    RESULTS.mkdir(exist_ok=True)
    subprocess.run(["make", "all"], cwd=ROOT, check=True)

    max_threads = os.cpu_count() or 32
    if os.environ.get("THREADS"):
        thread_counts = [int(x) for x in os.environ["THREADS"].split(",") if x]
    else:
        thread_counts = [1, 2, 4, 8, 16, max_threads]
    thread_counts = sorted(set(t for t in thread_counts if 1 <= t <= max_threads))
    schedules = [x for x in os.environ.get("SCHEDULES", "static,dynamic,guided").split(",") if x]
    repeats = int(os.environ.get("REPEATS", "1"))
    impls = [x for x in os.environ.get("IMPLS", "openmp,pthread").split(",") if x]
    total_runs = len(impls) * len(schedules) * len(thread_counts) * repeats

    raw_path = RESULTS / "heated_plate_raw.csv"
    summary_path = RESULTS / "heated_plate_summary.csv"
    print(f"results directory: {RESULTS}", flush=True)
    print(
        f"plan: impls={impls}, schedules={schedules}, "
        f"threads={thread_counts}, repeats={repeats}, total={total_runs}",
        flush=True,
    )
    fields = [
        "impl",
        "schedule",
        "M",
        "N",
        "threads",
        "chunk",
        "iterations",
        "diff",
        "time_sec",
        "checksum",
        "repeat",
    ]

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        run_id = 0
        for impl in impls:
            exe = f"./heated_plate_{impl}"
            for schedule in schedules:
                for threads in thread_counts:
                    for repeat in range(repeats):
                        run_id += 1
                        try:
                            row = run_once([exe, str(threads), schedule, "8"], run_id, total_runs)
                        except subprocess.TimeoutExpired:
                            print(
                                f"timeout after {RUN_TIMEOUT}s: {impl},{schedule},threads={threads}",
                                file=sys.stderr,
                                flush=True,
                            )
                            raise
                        row["repeat"] = repeat
                        writer.writerow(row)
                        f.flush()

    groups = {}
    with raw_path.open() as f:
        for row in csv.DictReader(f):
            key = (row["impl"], row["schedule"], row["threads"])
            groups.setdefault(key, []).append(float(row["time_sec"]))

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["impl", "schedule", "threads", "best_sec", "median_sec", "speedup_vs_openmp_1t"],
        )
        writer.writeheader()
        baseline = min(groups[("openmp", "static", "1")])
        for key in sorted(groups, key=lambda x: (x[0], x[1], int(x[2]))):
            times = groups[key]
            writer.writerow(
                {
                    "impl": key[0],
                    "schedule": key[1],
                    "threads": key[2],
                    "best_sec": f"{min(times):.6f}",
                    "median_sec": f"{statistics.median(times):.6f}",
                    "speedup_vs_openmp_1t": f"{baseline / min(times):.2f}",
                }
            )

    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()
