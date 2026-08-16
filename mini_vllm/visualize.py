"""
Visualizations of what the block manager and scheduler are actually doing,
step by step. Two forms: plain-text ASCII (cheap, good for logs/terminal
demos) and a matplotlib rendering (good for a README screenshot).
"""

from __future__ import annotations

from typing import List

from .block_manager import KVCacheManager
from .scheduler import Scheduler


def render_blocks_ascii(cache_manager: KVCacheManager, width: int = 40) -> str:
    """
    One character per physical block: '■' allocated, '□' free. Wraps every
    `width` characters so it's readable for large pools.
    """
    allocated_ids = set()
    for table in cache_manager._block_tables.values():  # noqa: SLF001 - read-only debug view
        allocated_ids.update(table.physical_blocks)

    chars = ["■" if i in allocated_ids else "□" for i in range(cache_manager.num_blocks)]
    lines = ["".join(chars[i : i + width]) for i in range(0, len(chars), width)]
    usage = cache_manager.usage()
    header = (
        f"blocks: {usage['used_blocks']}/{usage['num_blocks']} used "
        f"({usage['utilization_pct']}%)"
    )
    return header + "\n" + "\n".join(lines)


def render_running_waiting_ascii(scheduler: Scheduler, max_ids: int = 12) -> str:
    def fmt(ids: List[str]) -> str:
        shown = ids[:max_ids]
        text = ", ".join(shown)
        if len(ids) > max_ids:
            text += f", ... (+{len(ids) - max_ids} more)"
        return text or "(empty)"

    running_ids = [r.request_id for r in scheduler.running]
    waiting_ids = [r.request_id for r in scheduler.waiting]
    return f"running ({len(running_ids)}): {fmt(running_ids)}\nwaiting ({len(waiting_ids)}): {fmt(waiting_ids)}"


def render_step_ascii(cache_manager: KVCacheManager, scheduler: Scheduler, step_num: int) -> str:
    divider = "-" * 50
    return (
        f"{divider}\nIteration {step_num}\n{divider}\n"
        f"{render_running_waiting_ascii(scheduler)}\n\n"
        f"{render_blocks_ascii(cache_manager)}\n"
    )


def render_blocks_png(cache_manager: KVCacheManager, out_path: str, width: int = 32) -> None:
    """Saves a grid image of allocated (dark) vs free (light) blocks."""
    import matplotlib.pyplot as plt
    import numpy as np

    allocated_ids = set()
    for table in cache_manager._block_tables.values():  # noqa: SLF001
        allocated_ids.update(table.physical_blocks)

    n = cache_manager.num_blocks
    height = (n + width - 1) // width
    grid = np.zeros((height, width))
    for i in range(n):
        r, c = divmod(i, width)
        grid[r, c] = 1 if i in allocated_ids else 0

    cell_size = 0.5
    fig, ax = plt.subplots(figsize=(max(4, width * cell_size), max(2, height * cell_size)))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=1.4, aspect="equal")

    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])

    usage = cache_manager.usage()
    ax.set_title(
        f"KV cache blocks: {usage['used_blocks']}/{usage['num_blocks']} allocated "
        f"({usage['utilization_pct']}%)  |  ■ allocated  □ free",
        fontsize=11,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
