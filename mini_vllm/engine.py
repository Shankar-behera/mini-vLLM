"""
The engine wires the KV cache manager, scheduler, and (mock) model together
into the per-step loop a real inference server runs: schedule -> execute ->
sample -> update request state -> free finished requests' cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .block_manager import KVCacheManager
from .model import MockModel
from .request import FinishReason, Request
from .scheduler import Scheduler, SchedulerOutput


@dataclass
class StepResult:
    scheduler_output: SchedulerOutput
    step_latency_s: float
    finished_requests: List[Request] = field(default_factory=list)
    tokens_generated: int = 0


class LLMEngine:
    def __init__(
        self,
        num_blocks: int = 1024,
        block_size: int = 16,
        max_num_seqs: int = 16,
        max_num_batched_tokens: int = 2048,
        max_prefill_chunk_tokens: int = 512,
        hidden_dim: int = 256,
        vocab_size: int = 32000,
        model_seed: int = 0,
    ):
        self.cache_manager = KVCacheManager(num_blocks=num_blocks, block_size=block_size)
        self.scheduler = Scheduler(
            self.cache_manager,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
        )
        self.model = MockModel(hidden_dim=hidden_dim, vocab_size=vocab_size, seed=model_seed)
        self._requests: Dict[str, Request] = {}

    def add_request(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 64,
        stop_token_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> str:
        req = Request(
            request_id=request_id or Request.new_id(),
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            stop_token_id=stop_token_id,
        )
        self._requests[req.request_id] = req
        self.scheduler.add_request(req)
        return req.request_id

    def get_request(self, request_id: str) -> Request:
        return self._requests[request_id]

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    def step(self) -> StepResult:
        t0 = time.perf_counter()
        sched_out = self.scheduler.schedule()

        total_tokens = sched_out.num_batched_tokens
        hidden = self.model.forward_batch(total_tokens)

        units_needing_token = [
            u for u in sched_out.scheduled if (not u.is_prefill) or u.completes_prefill
        ]
        tokens_generated = 0
        finished: List[Request] = []

        if units_needing_token:
            sample_rows = hidden[: len(units_needing_token)]
            next_tokens = self.model.sample_next_tokens(sample_rows)
            for token_id, unit in zip(next_tokens, units_needing_token):
                request = unit.request
                request.append_output_token(int(token_id))
                tokens_generated += 1

                if request.should_stop():
                    request.mark_finished(request.finish_reason or FinishReason.MAX_TOKENS)
                    self.scheduler.remove_finished(request)
                    finished.append(request)

        step_latency = time.perf_counter() - t0
        return StepResult(
            scheduler_output=sched_out,
            step_latency_s=step_latency,
            finished_requests=finished,
            tokens_generated=tokens_generated,
        )

    def run_until_complete(self, max_steps: int = 100_000) -> List[Request]:
        """Convenience loop for offline/batch usage and benchmarking."""
        all_finished: List[Request] = []
        steps = 0
        while self.has_unfinished_requests() and steps < max_steps:
            result = self.step()
            all_finished.extend(result.finished_requests)
            steps += 1
        return all_finished
