#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import statistics
import subprocess
import sys


CSV_FIELDS = [
    "run_id",
    "started_at",
    "kernel",
    "m",
    "n",
    "k",
    "block_x",
    "block_y",
    "tile_k",
    "warmup",
    "repeats",
    "avg_ms",
    "min_ms",
    "max_ms",
    "gflops",
    "verified",
    "max_abs_error",
    "checksum",
]

MATRIX_SIZES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (1536, 1536, 1536),
    (2048, 2048, 2048),
    (512, 1024, 1536),
    (1024, 512, 2048),
    (1536, 1024, 512),
    (2048, 1536, 1024),
]

KERNELS_2D = ["global2d", "global2d_btrans", "shared_tiled"]
BLOCKS_2D = [(8, 8), (16, 16), (32, 8), (8, 32), (32, 16), (16, 32), (32, 32)]
BLOCKS_1D = [128, 256, 512, 1024]


def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


def run_cmd(cmd):
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def make_plan(quick):
    plan = []
    for m, n, k in MATRIX_SIZES:
        for bx, by in BLOCKS_2D:
            for kernel in KERNELS_2D:
                plan.append(
                    {
                        "kernel": kernel,
                        "m": m,
                        "n": n,
                        "k": k,
                        "block_x": bx,
                        "block_y": by,
                        "tile_k": bx,
                    }
                )
        for bx in BLOCKS_1D:
            plan.append(
                {
                    "kernel": "global1d",
                    "m": m,
                    "n": n,
                    "k": k,
                    "block_x": bx,
                    "block_y": 1,
                    "tile_k": bx,
                }
            )

    if quick:
        return plan[:18]
    return plan


def parse_program_csv(stdout):
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("benchmark produced no output")
    reader = csv.DictReader(lines)
    rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError(f"expected one CSV row, got {len(rows)}")
    return rows[0]


def run_one(binary, case, args):
    cmd = [
        binary,
        "--kernel",
        case["kernel"],
        "--m",
        str(case["m"]),
        "--n",
        str(case["n"]),
        "--k",
        str(case["k"]),
        "--block-x",
        str(case["block_x"]),
        "--block-y",
        str(case["block_y"]),
        "--tile-k",
        str(case["tile_k"]),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--verify-samples",
        str(args.verify_samples),
    ]
    result = run_cmd(cmd)
    return parse_program_csv(result.stdout)


def write_markdown(csv_path, md_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    numeric = ["avg_ms", "min_ms", "max_ms", "gflops", "max_abs_error", "checksum"]
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
        for key in ["m", "n", "k", "block_x", "block_y", "tile_k", "warmup", "repeats", "run_id"]:
            row[key] = int(row[key])

    best_by_size = {}
    for row in rows:
        size_key = (row["m"], row["n"], row["k"])
        current = best_by_size.get(size_key)
        if current is None or row["gflops"] > current["gflops"]:
            best_by_size[size_key] = row

    best_by_kernel = {}
    for row in rows:
        current = best_by_kernel.get(row["kernel"])
        if current is None or row["gflops"] > current["gflops"]:
            best_by_kernel[row["kernel"]] = row

    with open(md_path, "w") as f:
        f.write("# CUDA Matrix Multiplication Experiment Results\n\n")
        f.write(f"- Generated at: {now_iso()}\n")
        f.write(f"- Total cases: {len(rows)}\n")
        f.write(f"- Verified cases: {sum(1 for r in rows if r['verified'] == 'yes')}/{len(rows)}\n")
        if rows:
            f.write(f"- GFLOP/s range: {min(r['gflops'] for r in rows):.2f} - {max(r['gflops'] for r in rows):.2f}\n")
            f.write(f"- Median GFLOP/s: {statistics.median(r['gflops'] for r in rows):.2f}\n")
        f.write("\n")

        f.write("## Best Configuration Per Matrix Size\n\n")
        f.write("| m | n | k | kernel | block | tile_k | avg_ms | GFLOP/s | max_abs_error |\n")
        f.write("|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for key in sorted(best_by_size):
            row = best_by_size[key]
            f.write(
                f"| {row['m']} | {row['n']} | {row['k']} | {row['kernel']} | "
                f"{row['block_x']}x{row['block_y']} | {row['tile_k']} | "
                f"{row['avg_ms']:.4f} | {row['gflops']:.2f} | {row['max_abs_error']:.6g} |\n"
            )
        f.write("\n")

        f.write("## Best Configuration Per Kernel\n\n")
        f.write("| kernel | m | n | k | block | tile_k | avg_ms | GFLOP/s |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for kernel in sorted(best_by_kernel):
            row = best_by_kernel[kernel]
            f.write(
                f"| {row['kernel']} | {row['m']} | {row['n']} | {row['k']} | "
                f"{row['block_x']}x{row['block_y']} | {row['tile_k']} | "
                f"{row['avg_ms']:.4f} | {row['gflops']:.2f} |\n"
            )
        f.write("\n")

        f.write("## Full Data\n\n")
        f.write("| run | kernel | m | n | k | block | tile_k | avg_ms | min_ms | max_ms | GFLOP/s | verified | max_abs_error |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|\n")
        for row in rows:
            f.write(
                f"| {row['run_id']} | {row['kernel']} | {row['m']} | {row['n']} | {row['k']} | "
                f"{row['block_x']}x{row['block_y']} | {row['tile_k']} | "
                f"{row['avg_ms']:.4f} | {row['min_ms']:.4f} | {row['max_ms']:.4f} | "
                f"{row['gflops']:.2f} | {row['verified']} | {row['max_abs_error']:.6g} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Run CUDA matrix multiplication experiments.")
    parser.add_argument("--binary", default="./matrix_mul")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--verify-samples", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Run a short smoke-test subset.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.binary):
        raise FileNotFoundError(f"binary not found: {args.binary}")

    plan = make_plan(args.quick)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"matrix_mul_results_{stamp}.csv")
    md_path = os.path.join(args.output_dir, f"matrix_mul_results_{stamp}.md")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for idx, case in enumerate(plan, start=1):
            print(
                f"[run {idx}/{len(plan)}] {case['kernel']} "
                f"{case['m']}x{case['n']}x{case['k']} block={case['block_x']}x{case['block_y']}",
                flush=True,
            )
            row = run_one(args.binary, case, args)
            row["run_id"] = idx
            row["started_at"] = now_iso()
            writer.writerow({field: row[field] for field in CSV_FIELDS})
            f.flush()

    write_markdown(csv_path, md_path)
    print(f"[done] CSV: {csv_path}", flush=True)
    print(f"[done] Markdown: {md_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed: {' '.join(exc.cmd)}", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        sys.exit(exc.returncode)
