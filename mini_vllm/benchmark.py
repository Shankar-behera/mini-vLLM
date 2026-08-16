"""
Benchmarks that quantify what continuous batching and chunked prefill are
actually buying you, using the same engine and mock model for every scenario
so the only variable is scheduling policy.

Scenario A -- throughput: naive serial processing (max_num_seqs=1, so the
engine can only ever run one request at a time, which is what you get from a
request queue in front of a single-sequence generation loop) vs continuous
batching (max_num_seqs=N, many requests share the same iterations).

Scenario B -- head-of-line blocking: one very long prompt lands in the same
batch as several short, latency-sensitive requests. Compares monolithic
prefill (the whole prompt forced through in as few giant steps as possible)
against chunked prefill (the same prompt capped to small per-step slices),
and reports what happens to the short requests' latency while the long one
is being processed.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass
from typing import List

from .engine import LLMEngine


@dataclass
class RunStats:
    label: str
    wall_clock_s: float
    total_tokens_generated: int
    throughput_tokens_per_s: float
    latencies_ms: List[float]
    ttft_ms: List[float]
    num_steps: int

    @property
    def p50_latency_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = min(len(s) - 1, int(len(s) * 0.99))
        return s[idx]

    @property
    def mean_ttft_ms(self) -> float:
        return statistics.mean(self.ttft_ms) if self.ttft_ms else 0.0


def _make_workload(num_requests: int, seed: int = 42):
    rng = random.Random(seed)
    workload = []
    for _ in range(num_requests):
        prompt_len = rng.randint(20, 200)
        max_new_tokens = rng.randint(20, 80)
        workload.append((list(range(prompt_len)), max_new_tokens))
    return workload


def run_scenario(
    label: str,
    num_blocks: int,
    block_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_prefill_chunk_tokens: int,
    workload,
    prioritize_decode: bool = True,
) -> RunStats:
    engine = LLMEngine(
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_prefill_chunk_tokens=max_prefill_chunk_tokens,
    )
    engine.scheduler.prioritize_decode = prioritize_decode
    for prompt_ids, max_new in workload:
        engine.add_request(prompt_token_ids=prompt_ids, max_new_tokens=max_new)

    t0 = time.perf_counter()
    num_steps = 0
    total_tokens = 0
    while engine.has_unfinished_requests():
        result = engine.step()
        total_tokens += result.tokens_generated
        num_steps += 1
    wall_clock = time.perf_counter() - t0

    latencies = [
        r.total_latency * 1000 for r in engine._requests.values() if r.total_latency is not None
    ]
    ttfts = [
        r.time_to_first_token * 1000
        for r in engine._requests.values()
        if r.time_to_first_token is not None
    ]

    return RunStats(
        label=label,
        wall_clock_s=wall_clock,
        total_tokens_generated=total_tokens,
        throughput_tokens_per_s=total_tokens / wall_clock if wall_clock > 0 else 0.0,
        latencies_ms=latencies,
        ttft_ms=ttfts,
        num_steps=num_steps,
    )


def benchmark_naive_vs_continuous_batching(num_requests: int = 40) -> List[RunStats]:
    workload = _make_workload(num_requests)

    naive = run_scenario(
        label="naive (max_num_seqs=1)",
        num_blocks=4096,
        block_size=16,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        max_prefill_chunk_tokens=4096,
        workload=workload,
    )
    continuous = run_scenario(
        label="continuous batching (max_num_seqs=16)",
        num_blocks=4096,
        block_size=16,
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        max_prefill_chunk_tokens=512,
        workload=workload,
    )
    return [naive, continuous]


@dataclass
class HolBlockingStats:
    label: str
    long_request_ttft_ms: float
    long_request_latency_ms: float
    short_requests_mean_ttft_ms: float
    short_requests_max_ttft_ms: float
    short_requests_mean_latency_ms: float
    wall_clock_s: float


def run_hol_scenario(
    label: str,
    max_num_batched_tokens: int,
    max_prefill_chunk_tokens: int,
    workload,
) -> HolBlockingStats:
    engine = LLMEngine(
        num_blocks=4096,
        block_size=16,
        max_num_seqs=16,
        max_num_batched_tokens=max_num_batched_tokens,
        max_prefill_chunk_tokens=max_prefill_chunk_tokens,
    )
    for prompt_ids, max_new in workload:
        engine.add_request(prompt_token_ids=prompt_ids, max_new_tokens=max_new)

    t0 = time.perf_counter()
    while engine.has_unfinished_requests():
        engine.step()
    wall_clock = time.perf_counter() - t0

    reqs = list(engine._requests.values())
    long_req, short_reqs = reqs[0], reqs[1:]
    short_ttfts = [r.time_to_first_token * 1000 for r in short_reqs]
    short_latencies = [r.total_latency * 1000 for r in short_reqs]

    return HolBlockingStats(
        label=label,
        long_request_ttft_ms=long_req.time_to_first_token * 1000,
        long_request_latency_ms=long_req.total_latency * 1000,
        short_requests_mean_ttft_ms=statistics.mean(short_ttfts),
        short_requests_max_ttft_ms=max(short_ttfts),
        short_requests_mean_latency_ms=statistics.mean(short_latencies),
        wall_clock_s=wall_clock,
    )


def benchmark_chunked_vs_monolithic_prefill() -> List[HolBlockingStats]:
    """
    Mixes one 4000-token prompt with 15 short, decode-heavy requests, then
    compares an unbounded per-request prefill chunk (a request can eat the
    entire step's token budget) against a small, explicit chunk cap -- with
    everything else, including decode priority and the overall per-step
    token budget, held identical.

    Aggregate throughput/latency across all 16 requests together turns out
    to hide the actual effect (the long request's own numbers dominate or
    offset the short requests' numbers depending on which way you average),
    so this reports the long request and the short requests separately. The
    number that actually matters is short_requests_mean_ttft_ms: how long
    the 15 short, latency-sensitive requests sit in the FIFO waiting queue
    behind the long one before they're even admitted.
    """
    rng = random.Random(7)
    workload = [(list(range(4000)), 5)]
    for _ in range(15):
        workload.append((list(range(rng.randint(10, 30))), 40))

    unbounded = run_hol_scenario(
        label="unbounded prefill chunk",
        max_num_batched_tokens=512,
        max_prefill_chunk_tokens=512,
        workload=workload,
    )
    chunked = run_hol_scenario(
        label="chunked prefill (cap=64)",
        max_num_batched_tokens=512,
        max_prefill_chunk_tokens=64,
        workload=workload,
    )
    return [unbounded, chunked]


def print_hol_table(rows: List[HolBlockingStats]) -> None:
    header = (
        f"{'scenario':<28}{'long ttft(ms)':>15}{'long lat(ms)':>15}"
        f"{'short mean ttft(ms)':>21}{'short max ttft(ms)':>20}{'short mean lat(ms)':>20}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.label:<28}{r.long_request_ttft_ms:>15.1f}{r.long_request_latency_ms:>15.1f}"
            f"{r.short_requests_mean_ttft_ms:>21.1f}{r.short_requests_max_ttft_ms:>20.1f}"
            f"{r.short_requests_mean_latency_ms:>20.1f}"
        )


def print_table(rows: List[RunStats]) -> None:
    header = f"{'scenario':<40}{'wall(s)':>10}{'tok/s':>12}{'p50 lat(ms)':>14}{'p99 lat(ms)':>14}{'mean ttft(ms)':>16}{'steps':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.label:<40}{r.wall_clock_s:>10.3f}{r.throughput_tokens_per_s:>12.1f}"
            f"{r.p50_latency_ms:>14.2f}{r.p99_latency_ms:>14.2f}{r.mean_ttft_ms:>16.2f}{r.num_steps:>8}"
        )
