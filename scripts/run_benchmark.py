#!/usr/bin/env python3
"""
Run all benchmark scenarios, print a results table, and save a comparison
chart to benchmarks/results.png.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --requests 100
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mini_vllm.benchmark import (
    benchmark_chunked_vs_monolithic_prefill,
    benchmark_naive_vs_continuous_batching,
    print_hol_table,
    print_table,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--out-dir", default="benchmarks")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== Scenario A: naive serial vs. continuous batching ===\n")
    scenario_a = benchmark_naive_vs_continuous_batching(num_requests=args.requests)
    print_table(scenario_a)

    print("\n=== Scenario B: chunked prefill vs. head-of-line blocking ===\n")
    print("(one 4000-token prompt mixed with 15 short, decode-heavy requests)\n")
    scenario_b = benchmark_chunked_vs_monolithic_prefill()
    print_hol_table(scenario_b)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    labels_a = [r.label for r in scenario_a]
    axes[0].bar(labels_a, [r.throughput_tokens_per_s for r in scenario_a], color=["#c0392b", "#2980b9"])
    axes[0].set_title("Throughput: naive vs continuous batching")
    axes[0].set_ylabel("tokens / second")
    axes[0].tick_params(axis="x", rotation=15)

    labels_b = [r.label for r in scenario_b]
    axes[1].bar(
        labels_b,
        [r.short_requests_mean_ttft_ms for r in scenario_b],
        color=["#c0392b", "#2980b9"],
    )
    axes[1].set_title("Short requests' mean TTFT\nwhile a 4000-token prompt prefills")
    axes[1].set_ylabel("milliseconds")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(
        labels_b,
        [r.long_request_latency_ms for r in scenario_b],
        color=["#c0392b", "#2980b9"],
    )
    axes[2].set_title("Long request's own total latency\n(the trade-off)")
    axes[2].set_ylabel("milliseconds")
    axes[2].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "results.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    main()
