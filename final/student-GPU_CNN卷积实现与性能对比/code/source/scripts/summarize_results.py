#!/usr/bin/env python3
import csv
import datetime as dt
import pathlib
import statistics
import sys


def fnum(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main():
    if len(sys.argv) != 3:
        print("usage: summarize_results.py input.csv output.md", file=sys.stderr)
        return 2

    csv_path = pathlib.Path(sys.argv[1])
    md_path = pathlib.Path(sys.argv[2])

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("no rows in CSV")

    rows.sort(
        key=lambda r: (
            int(r["out_channels"]),
            int(r["input_size"]),
            int(r["stride"]),
            {"direct_cuda": 0, "im2col_cublas": 1, "cudnn": 2}.get(r["method"], 9),
        )
    )

    gpu = rows[0].get("gpu", "")
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_rows = []
    for r in rows:
        summary_rows.append(
            [
                r["out_channels"],
                r["input_size"],
                r["stride"],
                f'{r["out_h"]}x{r["out_w"]}',
                r["method"],
                fnum(r["mean_ms"], 4),
                fnum(r["std_ms"], 4),
                fnum(r["gflops"], 2),
                fnum(r["workspace_mb"], 2),
                fnum(r["max_abs_error"], 6),
                r["algorithm"],
            ]
        )

    grouped = {}
    for r in rows:
        grouped.setdefault((r["out_channels"], r["input_size"], r["stride"]), {})[r["method"]] = r

    speed_rows = []
    for key in sorted(grouped, key=lambda x: (int(x[0]), int(x[1]), int(x[2]))):
        methods = grouped[key]
        direct = float(methods["direct_cuda"]["mean_ms"]) if "direct_cuda" in methods else None
        im2col = float(methods["im2col_cublas"]["mean_ms"]) if "im2col_cublas" in methods else None
        cudnn = float(methods["cudnn"]["mean_ms"]) if "cudnn" in methods else None
        best_method = min(methods.values(), key=lambda r: float(r["mean_ms"]))["method"]
        speed_rows.append(
            [
                key[0],
                key[1],
                key[2],
                fnum(direct, 4) if direct is not None else "-",
                fnum(im2col, 4) if im2col is not None else "-",
                fnum(cudnn, 4) if cudnn is not None else "-",
                f"{direct / im2col:.2f}x" if direct and im2col else "-",
                f"{direct / cudnn:.2f}x" if direct and cudnn else "-",
                best_method,
            ]
        )

    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(float(r["gflops"]))
    aggregate_rows = []
    for method in sorted(by_method):
        vals = by_method[method]
        aggregate_rows.append(
            [
                method,
                len(vals),
                f"{statistics.mean(vals):.2f}",
                f"{max(vals):.2f}",
            ]
        )

    sample_rows = []
    for r in rows:
        if r["out_channels"] == "1":
            sample_rows.append(
                [
                    r["input_size"],
                    r["stride"],
                    r["method"],
                    fnum(r["checksum"], 6),
                    fnum(r["sample0"], 6),
                    fnum(r["sample1"], 6),
                    fnum(r["sample2"], 6),
                    fnum(r["sample3"], 6),
                ]
            )

    text = []
    text.append("# CNN Convolution CUDA Experiment Results")
    text.append("")
    text.append(f"- Generated at: {now}")
    text.append(f"- GPU: {gpu}")
    text.append("- Input: NCHW, batch=1, input channels=3, kernel=3x3, bias=0")
    text.append("- Padding: 1 on both height and width; output size follows `floor((H + 2P - K) / stride) + 1`")
    text.append("- Methods: direct CUDA sliding window, im2col + cuBLAS SGEMM, cuDNN convolution")
    text.append("")
    text.append("## Method Timing Table")
    text.append("")
    text.append(
        markdown_table(
            [
                "out_ch",
                "input",
                "stride",
                "output",
                "method",
                "mean_ms",
                "std_ms",
                "GFLOP/s",
                "workspace_MB",
                "max_abs_err",
                "algorithm",
            ],
            summary_rows,
        )
    )
    text.append("")
    text.append("## Same-Shape Speed Comparison")
    text.append("")
    text.append(
        markdown_table(
            [
                "out_ch",
                "input",
                "stride",
                "direct_ms",
                "im2col_ms",
                "cudnn_ms",
                "direct/im2col",
                "direct/cudnn",
                "best",
            ],
            speed_rows,
        )
    )
    text.append("")
    text.append("## Aggregate Throughput")
    text.append("")
    text.append(markdown_table(["method", "cases", "mean_GFLOP/s", "max_GFLOP/s"], aggregate_rows))
    text.append("")
    text.append("## Assignment-Scale Output Samples (`out_channels=1`)")
    text.append("")
    text.append(
        markdown_table(
            ["input", "stride", "method", "checksum", "sample0", "sample1", "sample2", "sample3"],
            sample_rows,
        )
    )
    text.append("")
    text.append("## Notes for Report")
    text.append("")
    text.append(
        "- `out_channels=1` matches the minimal single-filter requirement in the README. "
        "`out_channels=16/64/128` are additional pressure tests for better GEMM/cuDNN utilization."
    )
    text.append(
        "- The `out_channels=128, input=4096, stride=1` pressure case is intentionally skipped by "
        "the runner because its output tensor has `2^31` elements, which cuDNN's legacy forward API "
        "rejects on this setup."
    )
    text.append(
        "- `max_abs_err` compares im2col/cuDNN output with the direct CUDA output. Small nonzero "
        "differences are expected when Tensor Core/TF32 paths are selected by libraries."
    )

    md_path.write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
