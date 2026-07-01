#!/usr/bin/env python3
"""Generate reproducible shortest-path query files from a graph CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def read_vertices(graph_path: Path) -> list[int]:
    vertices: set[int] = set()
    with graph_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vertices.add(int(row["source"]))
            vertices.add(int(row["target"]))
    return sorted(vertices)


def write_all_pairs(vertices: list[int], output_path: Path, include_self: bool) -> int:
    count = 0
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for source in vertices:
            for target in vertices:
                if not include_self and source == target:
                    continue
                writer.writerow([source, target])
                count += 1
    return count


def write_random_pairs(vertices: list[int], output_path: Path, count: int, seed: int) -> int:
    rng = random.Random(seed)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for _ in range(count):
            source = rng.choice(vertices)
            target = rng.choice(vertices)
            while target == source and len(vertices) > 1:
                target = rng.choice(vertices)
            writer.writerow([source, target])
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["all-pairs", "random"], default="all-pairs")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--include-self", action="store_true")
    args = parser.parse_args()

    vertices = read_vertices(args.graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "all-pairs":
        count = write_all_pairs(vertices, args.output, args.include_self)
    else:
        count = write_random_pairs(vertices, args.output, args.count, args.seed)
    print(f"graph={args.graph} vertices={len(vertices)} output={args.output} queries={count}")


if __name__ == "__main__":
    main()
