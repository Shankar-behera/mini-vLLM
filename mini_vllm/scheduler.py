"""
Iteration-level scheduler with continuous batching and chunked prefill.

Every call to schedule() represents one "engine step" (one forward pass in a
real server). On each step it:

  1. Runs one decode token for every already-running request that has
     finished prefill -- decode always gets first claim on the token budget
     so long-running generations are never starved by a newly arrived long
     prompt (that's the head-of-line blocking problem chunked prefill exists
     to solve).
  2. Spends whatever token budget is left continuing prefill for requests
     that are still mid-prompt, in bounded-size chunks.
  3. Admits new waiting requests into the running batch if there's spare
     token budget, sequence slots, and physical KV cache blocks for at least
     a partial first chunk.

Design note: unlike a real GPU-backed engine, this scheduler commits to a
scheduling decision and applies it (block allocation + token bookkeeping) in
the same call, since the mock model behind it can't fail a forward pass.
A real engine splits "schedule" from "execute" because GPU OOM at runtime is
possible; here that split would just be ceremony.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from .block_manager import KVCacheManager
from .request import Request


@dataclass
class ScheduledUnit:
    """One request's slice of work for the current step."""

    request: Request
    num_tokens: int
    is_prefill: bool
    completes_prefill: bool  # True if this chunk finishes the prompt (next step decodes)


@dataclass
class SchedulerOutput:
    scheduled: List[ScheduledUnit] = field(default_factory=list)
    preempted_request_ids: List[str] = field(default_factory=list)
    num_batched_tokens: int = 0
    num_running: int = 0
    num_waiting: int = 0
    cache_usage: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(
        self,
        cache_manager: KVCacheManager,
        max_num_seqs: int = 16,
        max_num_batched_tokens: int = 2048,
        max_prefill_chunk_tokens: int = 512,
        prioritize_decode: bool = True,
    ):
        self.cache_manager = cache_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_prefill_chunk_tokens = max_prefill_chunk_tokens
        # Real production schedulers (and this simulator's default) always let
        # decode claim the token budget first, precisely so an arriving
        # prompt can never fully stall requests that are already generating.
        # Setting this False reproduces the pre-chunked-prefill failure mode
        # (prefill goes first and can eat the whole step) purely so the
        # benchmark can demonstrate why the priority + chunk-size combination
        # matters -- it is not meant to be used in the live API.
        self.prioritize_decode = prioritize_decode

        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def remove_finished(self, request: Request) -> None:
        """Called by the engine once a request's generation has stopped."""
        if request in self.running:
            self.running.remove(request)
        self.cache_manager.free(request.request_id)

    # -- internal helpers -------------------------------------------------

    def _select_preemption_victim(self, protect: Request) -> Optional[Request]:
        """
        Pick a running request to evict when the cache is full and a
        higher-priority request needs a block. Prefers the most recently
        admitted request, matching vLLM's default recompute preemption
        ordering (last in, first out).
        """
        for candidate in reversed(self.running):
            if candidate.request_id != protect.request_id and self.cache_manager.has_request(
                candidate.request_id
            ):
                return candidate
        return None

    def _preempt(self, request: Request) -> None:
        """
        Evict a running request: free its blocks and push it back to the
        front of the waiting queue for full recompute later.

        Simplification vs. real vLLM: we discard KV progress for the
        original prompt only and reset num_computed_tokens to 0. Already
        generated output tokens are kept as-is; a fully faithful recompute
        policy would fold them back into the context to re-prefill, which is
        unnecessary complexity for what this simulator is trying to show.
        """
        self.cache_manager.free(request.request_id)
        request.num_computed_tokens = 0
        request.status = request.status.__class__.WAITING
        self.running.remove(request)
        self.waiting.appendleft(request)

    def _try_fit_decode_step(self, request: Request, preempted_ids: List[str]) -> bool:
        while not self.cache_manager.can_append_tokens(request.request_id, 1):
            victim = self._select_preemption_victim(protect=request)
            if victim is None:
                return False
            self._preempt(victim)
            preempted_ids.append(victim.request_id)
        return True

    # -- main entry point ---------------------------------------------------

    def _schedule_decode_phase(
        self, token_budget: int, scheduled: List[ScheduledUnit], preempted_ids: List[str]
    ) -> int:
        decode_requests = [r for r in self.running if r.is_prefill_complete]
        for request in decode_requests:
            if token_budget <= 0:
                break
            if request not in self.running:
                continue  # evicted earlier in this same step by another request's preemption
            if not self._try_fit_decode_step(request, preempted_ids):
                continue  # cache is full even after preempting everything possible
            self.cache_manager.append_token(request.request_id)
            scheduled.append(
                ScheduledUnit(request, num_tokens=1, is_prefill=False, completes_prefill=False)
            )
            token_budget -= 1
        return token_budget

    def _schedule_prefill_phase(self, token_budget: int, scheduled: List[ScheduledUnit]) -> int:
        # continue in-flight prefills, chunked
        prefill_requests = [r for r in self.running if not r.is_prefill_complete]
        for request in prefill_requests:
            if token_budget <= 0:
                break
            if request not in self.running:
                continue  # evicted earlier in this same step
            desired = min(
                self.max_prefill_chunk_tokens, request.remaining_prompt_tokens, token_budget
            )
            if desired <= 0:
                continue
            chunk = self.cache_manager.max_appendable_tokens(request.request_id, desired)
            if chunk <= 0:
                continue
            self.cache_manager.append_chunk(request.request_id, chunk)
            request.record_prefill_progress(chunk)
            completes = request.is_prefill_complete
            scheduled.append(
                ScheduledUnit(request, num_tokens=chunk, is_prefill=True, completes_prefill=completes)
            )
            token_budget -= chunk

        # admit new requests from the waiting queue
        while self.waiting and token_budget > 0 and len(self.running) < self.max_num_seqs:
            candidate = self.waiting[0]
            desired = min(self.max_prefill_chunk_tokens, candidate.prompt_len, token_budget)
            if desired <= 0:
                break
            fittable_from_free = self.cache_manager.num_free_blocks * self.cache_manager.block_size
            chunk = min(desired, fittable_from_free)
            if chunk <= 0:
                break  # cache genuinely full, nothing more we can admit this step
            self.waiting.popleft()
            self.cache_manager.allocate(candidate.request_id, chunk)
            candidate.record_prefill_progress(chunk)
            candidate.status = candidate.status.__class__.RUNNING
            self.running.append(candidate)
            completes = candidate.is_prefill_complete
            scheduled.append(
                ScheduledUnit(candidate, num_tokens=chunk, is_prefill=True, completes_prefill=completes)
            )
            token_budget -= chunk
        return token_budget

    # -- main entry point ---------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        scheduled: List[ScheduledUnit] = []
        preempted_ids: List[str] = []
        token_budget = self.max_num_batched_tokens

        if self.prioritize_decode:
            # Default, production-like ordering: decode always gets served
            # first, so an arriving prompt can never fully stall requests
            # that are already generating -- this is what makes chunking the
            # prefill (rather than just capping it) matter: even a request
            # that couldn't be prioritized still only ever nibbles a bounded
            # slice of whatever budget decode left behind.
            token_budget = self._schedule_decode_phase(token_budget, scheduled, preempted_ids)
            token_budget = self._schedule_prefill_phase(token_budget, scheduled)
        else:
            # Reproduces the pre-chunked-prefill failure mode for the
            # benchmark: prefill claims the budget first, so a large,
            # unchunked prompt can consume the entire step and leave nothing
            # for decode -- a full stall for every other in-flight request
            # until that one step (however long it takes) finishes.
            token_budget = self._schedule_prefill_phase(token_budget, scheduled)
            token_budget = self._schedule_decode_phase(token_budget, scheduled, preempted_ids)

        return SchedulerOutput(
            scheduled=scheduled,
            preempted_request_ids=preempted_ids,
            num_batched_tokens=self.max_num_batched_tokens - token_budget,
            num_running=len(self.running),
            num_waiting=len(self.waiting),
            cache_usage=self.cache_manager.usage(),
        )
