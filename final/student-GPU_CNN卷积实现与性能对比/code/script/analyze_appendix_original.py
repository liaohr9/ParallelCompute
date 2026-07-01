#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = ROOT.parent
RESULTS = ROOT / "results"
IMG = ROOT / "img"


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    return float(row[key])


def inum(row, key):
    return int(row[key])


def shape_key(row):
    return (inum(row, "input_size"), inum(row, "out_channels"), inum(row, "stride"))


def tex_escape(value):
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def geo_mean(values):
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_main_best():
    main_path = MAIN_ROOT / "results" / "experiment_results.csv"
    if not main_path.exists():
        return {}
    best = {}
    for row in read_rows(main_path):
        key = shape_key(row)
        mean_ms = fnum(row, "mean_ms")
        current = best.get(key)
        if current is None or mean_ms < current["mean_ms"]:
            best[key] = {
                "method": row["method"],
                "mean_ms": mean_ms,
                "gflops": fnum(row, "gflops"),
            }
    return best


def method_summary(rows, fastest_by_shape):
    direct_by_shape = {
        shape_key(r): fnum(r, "mean_ms")
        for r in rows
        if r["method"] == "direct_baseline"
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    fastest_counts = Counter(v["method"] for v in fastest_by_shape.values())
    out = []
    for method, items in sorted(grouped.items()):
        speedups = []
        for row in items:
            base = direct_by_shape[shape_key(row)]
            speedups.append(base / fnum(row, "mean_ms"))
        out.append(
            {
                "method": method,
                "optimization": items[0]["optimization"],
                "cases": len(items),
                "mean_gflops": statistics.mean(fnum(r, "gflops") for r in items),
                "median_gflops": statistics.median(fnum(r, "gflops") for r in items),
                "max_gflops": max(fnum(r, "gflops") for r in items),
                "geo_speedup_vs_direct": geo_mean(speedups),
                "fastest_count": fastest_counts[method],
                "max_abs_error": max(fnum(r, "max_abs_error") for r in items),
                "max_workspace_mb": max(fnum(r, "workspace_mb") for r in items),
            }
        )
    return out


def fastest_rows(rows):
    by_shape = defaultdict(list)
    for row in rows:
        by_shape[shape_key(row)].append(row)
    fastest = {}
    for key, items in by_shape.items():
        fastest[key] = min(items, key=lambda r: fnum(r, "mean_ms"))
    return fastest


def selector_summary(rows, fastest_by_shape, main_best):
    by_shape_method = {(shape_key(r), r["method"]): r for r in rows}
    direct_by_shape = {
        shape_key(r): fnum(r, "mean_ms")
        for r in rows
        if r["method"] == "direct_baseline"
    }

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (inum(row, "out_channels"), inum(row, "stride"))
        grouped[key][row["method"]].append(fnum(row, "mean_ms"))

    rule = {}
    rule_rows = []
    for key, methods in sorted(grouped.items()):
        method_scores = {m: geo_mean(v) for m, v in methods.items()}
        selected = min(method_scores, key=method_scores.get)
        rule[key] = selected
        rule_rows.append(
            {
                "out_channels": key[0],
                "stride": key[1],
                "selected_method": selected,
                "geo_mean_ms": method_scores[selected],
            }
        )

    write_csv(
        RESULTS / "auto_selector_rule.csv",
        ["out_channels", "stride", "selected_method", "geo_mean_ms"],
        rule_rows,
    )

    oracle_speedups = []
    rule_speedups = []
    main_oracle_speedups = []
    main_rule_speedups = []
    for key, best in fastest_by_shape.items():
        direct = direct_by_shape[key]
        oracle_speedups.append(direct / fnum(best, "mean_ms"))
        selected = rule[(key[1], key[2])]
        selected_row = by_shape_method[(key, selected)]
        rule_speedups.append(direct / fnum(selected_row, "mean_ms"))
        if key in main_best:
            main_oracle_speedups.append(main_best[key]["mean_ms"] / fnum(best, "mean_ms"))
            main_rule_speedups.append(main_best[key]["mean_ms"] / fnum(selected_row, "mean_ms"))

    return [
        {
            "selector": "oracle_best",
            "description": "逐 shape 选择实测最快优化方法",
            "cases": len(fastest_by_shape),
            "geo_speedup_vs_direct": geo_mean(oracle_speedups),
            "geo_speedup_vs_main_best": geo_mean(main_oracle_speedups) if main_oracle_speedups else 0.0,
        },
        {
            "selector": "profile_rule",
            "description": "按 out_channels 与 stride 选择分组最快方法",
            "cases": len(fastest_by_shape),
            "geo_speedup_vs_direct": geo_mean(rule_speedups),
            "geo_speedup_vs_main_best": geo_mean(main_rule_speedups) if main_rule_speedups else 0.0,
        },
    ]


def compare_with_main(fastest_by_shape, main_best):
    rows = []
    for key, app_best in fastest_by_shape.items():
        if key not in main_best:
            continue
        main = main_best[key]
        app_ms = fnum(app_best, "mean_ms")
        speedup = main["mean_ms"] / app_ms
        if speedup > 1.02:
            rows.append(
                {
                    "input_size": key[0],
                    "out_channels": key[1],
                    "stride": key[2],
                    "main_best_method": main["method"],
                    "main_best_ms": main["mean_ms"],
                    "appendix_best_method": app_best["method"],
                    "appendix_best_ms": app_ms,
                    "speedup": speedup,
                }
            )
    rows.sort(key=lambda r: r["speedup"], reverse=True)
    return rows


def write_latex_tables(summary, selector, improvements):
    with open(RESULTS / "generated_tables.tex", "w") as f:
        f.write("% Auto-generated by scripts/analyze_appendix.py\n")
        f.write("\\begin{table}[H]\n\\centering\n\\caption{附录优化方法总体统计}\n")
        f.write("\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
        f.write("方法 & case数 & 平均GFLOP/s & 中位GFLOP/s & 最高GFLOP/s & 相对direct几何加速 & 最快次数 & 最大误差 \\\\\n")
        f.write("\\midrule\n")
        for row in summary:
            f.write(
                f"{tex_escape(row['method'])} & {row['cases']} & {row['mean_gflops']:.2f} & "
                f"{row['median_gflops']:.2f} & {row['max_gflops']:.2f} & "
                f"{row['geo_speedup_vs_direct']:.3f}x & {row['fastest_count']} & "
                f"{row['max_abs_error']:.6g} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n\n")

        f.write("\\begin{table}[H]\n\\centering\n\\caption{自动选择策略汇总}\n")
        f.write("\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{llrrr}\n\\toprule\n")
        f.write("选择器 & 说明 & case数 & 相对direct几何加速 & 相对主实验最优几何加速 \\\\\n")
        f.write("\\midrule\n")
        for row in selector:
            main_speed = "-" if row["geo_speedup_vs_main_best"] == 0 else f"{row['geo_speedup_vs_main_best']:.3f}x"
            f.write(
                f"{tex_escape(row['selector'])} & {tex_escape(row['description'])} & {row['cases']} & "
                f"{row['geo_speedup_vs_direct']:.3f}x & {main_speed} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n\n")

        f.write("\\begin{table}[H]\n\\centering\n\\caption{相对主实验三方法最优结果仍有提升的代表 case}\n")
        f.write("\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{rrrllrrr}\n\\toprule\n")
        f.write("输入 & out\\_ch & stride & 主实验最优 & 附录最优 & 主实验/ms & 附录/ms & 加速比 \\\\\n")
        f.write("\\midrule\n")
        if improvements:
            for row in improvements[:12]:
                f.write(
                    f"{row['input_size']} & {row['out_channels']} & {row['stride']} & "
                    f"{tex_escape(row['main_best_method'])} & {tex_escape(row['appendix_best_method'])} & "
                    f"{row['main_best_ms']:.4f} & {row['appendix_best_ms']:.4f} & {row['speedup']:.3f}x \\\\\n"
                )
        else:
            f.write("\\multicolumn{8}{c}{未发现超过主实验三方法最优值 2\\% 以上的 case} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")


def make_plots(rows, summary, fastest_by_shape, improvements):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: matplotlib unavailable: {exc}", file=sys.stderr)
        return

    IMG.mkdir(exist_ok=True)

    methods = [r["method"] for r in summary]
    means = [r["mean_gflops"] for r in summary]
    plt.figure(figsize=(9, 4.8))
    plt.bar(methods, means, color="#3b82f6")
    plt.ylabel("Average GFLOP/s")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(IMG / "appendix_mean_gflops.png", dpi=180)
    plt.close()

    counts = Counter(r["method"] for r in fastest_by_shape.values())
    plt.figure(figsize=(8, 4.8))
    plt.bar(list(counts.keys()), list(counts.values()), color="#10b981")
    plt.ylabel("Fastest case count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(IMG / "appendix_fastest_counts.png", dpi=180)
    plt.close()

    direct_by_shape = {shape_key(r): fnum(r, "mean_ms") for r in rows if r["method"] == "direct_baseline"}
    by_out = defaultdict(lambda: defaultdict(list))
    for row in rows:
        out_ch = inum(row, "out_channels")
        speed = direct_by_shape[shape_key(row)] / fnum(row, "mean_ms")
        by_out[out_ch][row["method"]].append(speed)
    out_channels = sorted(by_out)
    methods = sorted({r["method"] for r in rows})
    x = list(range(len(out_channels)))
    width = 0.12
    plt.figure(figsize=(10, 5.2))
    for i, method in enumerate(methods):
        vals = [geo_mean(by_out[oc][method]) for oc in out_channels]
        offsets = [v + (i - len(methods) / 2) * width for v in x]
        plt.bar(offsets, vals, width=width, label=method)
    plt.axhline(1.0, color="black", linewidth=0.8)
    plt.ylabel("Geomean speedup vs direct_baseline")
    plt.xlabel("out_channels")
    plt.xticks(x, [str(v) for v in out_channels])
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(IMG / "appendix_speedup_by_out_channels.png", dpi=180)
    plt.close()

    if improvements:
        top = improvements[:10]
        labels = [f"{r['input_size']}/oc{r['out_channels']}/s{r['stride']}" for r in top]
        vals = [r["speedup"] for r in top]
        plt.figure(figsize=(9, 4.8))
        plt.bar(labels, vals, color="#f97316")
        plt.ylabel("Speedup vs main best")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(IMG / "appendix_main_improvements.png", dpi=180)
        plt.close()


def main():
    if len(sys.argv) != 2:
        print("usage: analyze_appendix.py <appendix_optimization.csv>", file=sys.stderr)
        return 2
    RESULTS.mkdir(exist_ok=True)
    rows = read_rows(Path(sys.argv[1]))
    fastest = fastest_rows(rows)
    summary = method_summary(rows, fastest)
    main_best = load_main_best()
    selector = selector_summary(rows, fastest, main_best)
    improvements = compare_with_main(fastest, main_best)

    write_csv(
        RESULTS / "summary_by_method.csv",
        [
            "method",
            "optimization",
            "cases",
            "mean_gflops",
            "median_gflops",
            "max_gflops",
            "geo_speedup_vs_direct",
            "fastest_count",
            "max_abs_error",
            "max_workspace_mb",
        ],
        summary,
    )
    write_csv(
        RESULTS / "improvements_vs_main_best.csv",
        [
            "input_size",
            "out_channels",
            "stride",
            "main_best_method",
            "main_best_ms",
            "appendix_best_method",
            "appendix_best_ms",
            "speedup",
        ],
        improvements,
    )
    write_csv(
        RESULTS / "auto_selector_summary.csv",
        [
            "selector",
            "description",
            "cases",
            "geo_speedup_vs_direct",
            "geo_speedup_vs_main_best",
        ],
        selector,
    )
    write_latex_tables(summary, selector, improvements)
    make_plots(rows, summary, fastest, improvements)
    print(f"loaded {len(rows)} rows, {len(fastest)} shapes")


if __name__ == "__main__":
    raise SystemExit(main())
