"""
A second, much simpler "engine" that produces real text from a real model
(anything pulled into a locally running Ollama, e.g. phi3:mini) instead of
the mock model the core simulator uses.

This is intentionally NOT wired into Scheduler/KVCacheManager. Those classes
simulate token-level, block-mapped continuous batching against a model we
fully control the forward pass of -- that's the actual PagedAttention/
continuous-batching mechanics this project is demonstrating. Ollama doesn't
expose that level of control (no per-token, multi-sequence batched forward
pass hook), so pretending to run it "through the scheduler" would just be
theater around a black-box HTTP call.

What this class does control honestly: how many concurrent generations are
allowed to run against Ollama at once (max_num_seqs, enforced with a
semaphore), and real per-request timing (TTFT, latency, tokens/sec) against
real streamed output. Requests beyond the concurrency limit genuinely queue
here before Ollama ever sees them.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Optional

from .backends.ollama_backend import OllamaBackend
from .request import Request as _Request  # reuse the id generator only


@dataclass
class RealRequestResult:
    request_id: str
    generated_text: str
    finish_reason: str
    time_to_first_token_ms: Optional[float]
    total_latency_ms: float
    tokens_generated: int
    tokens_per_second: Optional[float]


@dataclass
class StreamEvent:
    """One item from generate_stream: either a token piece or the final summary."""

    type: str  # "token" | "done"
    text: Optional[str] = None
    finish_reason: Optional[str] = None
    tokens_generated: Optional[int] = None
    time_to_first_token_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None


@dataclass
class _QueueStats:
    num_running: int = 0
    num_waiting: int = 0


class RealLLMEngine:
    def __init__(
        self,
        max_num_seqs: int = 8,
        model: str = "phi3:mini",
        host: str = "http://localhost:11434",
    ):
        self.max_num_seqs = max_num_seqs
        self.model = model
        self.backend = OllamaBackend(host=host)
        self._semaphore = asyncio.Semaphore(max_num_seqs)
        self._stats = _QueueStats()
        self._stats_lock = asyncio.Lock()

    async def stats(self) -> Dict[str, int]:
        async with self._stats_lock:
            return {"num_running": self._stats.num_running, "num_waiting": self._stats.num_waiting}

    async def _admit(self) -> None:
        async with self._stats_lock:
            self._stats.num_waiting += 1
        await self._semaphore.acquire()
        async with self._stats_lock:
            self._stats.num_waiting -= 1
            self._stats.num_running += 1

    async def _release(self) -> None:
        self._semaphore.release()
        async with self._stats_lock:
            self._stats.num_running -= 1

    async def generate(self, prompt: str, max_tokens: int = 128) -> RealRequestResult:
        request_id = _Request.new_id()
        t0 = time.perf_counter()
        await self._admit()
        try:
            text_parts = []
            first_token_at: Optional[float] = None
            num_pieces = 0
            finish_reason = "length"

            async for chunk in self.backend.astream(prompt, self.model, max_tokens):
                piece = chunk.get("response", "")
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(piece)
                    num_pieces += 1
                if chunk.get("done"):
                    finish_reason = chunk.get("done_reason", "stop")
                    break

            t1 = time.perf_counter()
            ttft_ms = (first_token_at - t0) * 1000 if first_token_at else None
            latency_ms = (t1 - t0) * 1000
            tps = num_pieces / (t1 - t0) if (t1 - t0) > 0 else None

            return RealRequestResult(
                request_id=request_id,
                generated_text="".join(text_parts),
                finish_reason=finish_reason,
                time_to_first_token_ms=ttft_ms,
                total_latency_ms=latency_ms,
                tokens_generated=num_pieces,
                tokens_per_second=tps,
            )
        finally:
            await self._release()

    async def generate_stream(
        self, prompt: str, max_tokens: int = 128
    ) -> AsyncIterator[StreamEvent]:
        """
        Yields a StreamEvent per generated piece, then a final type="done"
        event carrying finish_reason and stats -- without this, a caller has
        no way to tell a natural stop from hitting max_tokens mid-sentence.
        """
        t0 = time.perf_counter()
        await self._admit()
        try:
            first_token_at: Optional[float] = None
            num_pieces = 0
            finish_reason = "length"

            async for chunk in self.backend.astream(prompt, self.model, max_tokens):
                piece = chunk.get("response", "")
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    num_pieces += 1
                    yield StreamEvent(type="token", text=piece)
                if chunk.get("done"):
                    finish_reason = chunk.get("done_reason", "stop")
                    break

            t1 = time.perf_counter()
            ttft_ms = (first_token_at - t0) * 1000 if first_token_at else None
            yield StreamEvent(
                type="done",
                finish_reason=finish_reason,
                tokens_generated=num_pieces,
                time_to_first_token_ms=ttft_ms,
                total_latency_ms=(t1 - t0) * 1000,
            )
        finally:
            await self._release()