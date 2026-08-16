#!/usr/bin/env python3
"""
Runs a small, deliberately visualization-friendly workload and produces:

  - benchmarks/schedule_timeline.png   a Gantt-style chart: which requests
                                        are running/waiting each iteration
  - benchmarks/block_allocation.png    a snapshot of the KV cache block grid
                                        partway through the run
  - ASCII versions of both, printed to stdout

Usage:
    python scripts/visualize_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mini_vllm.engine import LLMEngine
from mini_vllm.visualize import render_blocks_png, render_step_ascii


STATUS_COLORS = {
    "prefill": "#f39c12",
    "decode": "#2980b9",
    "waiting": "#d5d8dc",
    "finished": "#ffffff",
}


def main():
    os.makedirs("benchmarks", exist_ok=True)

    engine = LLMEngine(
        num_blocks=40,
        block_size=8,
        max_num_seqs=3,
        max_num_batched_tokens=48,
        max_prefill_chunk_tokens=16,
    )

    workload = [
        (list(range(20)), 6),   # A: medium prompt
        (list(range(6)), 8),    # B: short prompt, longer generation
        (list(range(40)), 4),   # C: long prompt
        (list(range(5)), 5),    # D: arrives, has to wait (max_num_seqs=3)
        (list(range(10)), 5),   # E: also has to wait
    ]
    labels = ["A", "B", "C", "D", "E"]
    request_ids = [
        engine.add_request(prompt_token_ids=p, max_new_tokens=m, request_id=label)
        for label, (p, m) in zip(labels, workload)
    ]

    history = []  # list of {request_id: status} per iteration
    block_snapshot_step = None
    block_snapshot_ascii = None

    step_num = 0
    while engine.has_unfinished_requests() and step_num < 30:
        step_num += 1
        result = engine.step()

        status_this_step = {rid: "waiting" for rid in request_ids}
        for unit in result.scheduler_output.scheduled:
            status_this_step[unit.request.request_id] = "prefill" if unit.is_prefill else "decode"
        for req in engine._requests.values():
            if req.status.value == "finished" and req.request_id not in [
                u.request.request_id for u in result.scheduler_output.scheduled
            ]:
                status_this_step[req.request_id] = "finished"

        history.append(status_this_step)

        print(render_step_ascii(engine.cache_manager, engine.scheduler, step_num))

        if block_snapshot_step is None and engine.cache_manager.usage()["utilization_pct"] > 20:
            block_snapshot_step = step_num
            block_snapshot_ascii = render_blocks_png  # placeholder, real render below
            render_blocks_png(engine.cache_manager, "benchmarks/block_allocation.png", width=10)

    # once-finished requests should show as "finished" for all subsequent steps too
    finished_by = {}
    for i, snap in enumerate(history):
        for rid, status in snap.items():
            if status == "finished" and rid not in finished_by:
                finished_by[rid] = i
    for i, snap in enumerate(history):
        for rid in request_ids:
            if rid in finished_by and finished_by[rid] < i:
                snap[rid] = "finished"

    # --- Gantt-style timeline chart ---
    fig, ax = plt.subplots(figsize=(max(6, len(history) * 0.35), 3))
    for row, rid in enumerate(request_ids):
        for col, snap in enumerate(history):
            status = snap.get(rid, "waiting")
            color = STATUS_COLORS[status]
            ax.add_patch(
                plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor="white", linewidth=0.5)
            )
    ax.set_xlim(0, len(history))
    ax.set_ylim(0, len(request_ids))
    ax.set_yticks([i + 0.5 for i in range(len(request_ids))])
    ax.set_yticklabels([f"req {rid}" for rid in request_ids])
    ax.set_xlabel("scheduler iteration")
    ax.set_title("Continuous batching: per-iteration request status")
    ax.invert_yaxis()

    legend_handles = [mpatches.Patch(color=c, label=s) for s, c in STATUS_COLORS.items()]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)

    fig.tight_layout()
    fig.savefig("benchmarks/schedule_timeline.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved benchmarks/schedule_timeline.png ({len(history)} iterations)")
    if block_snapshot_step:
        print(f"Saved benchmarks/block_allocation.png (snapshot at iteration {block_snapshot_step})")


if __name__ == "__main__":
    main()
