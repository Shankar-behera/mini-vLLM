"""Request state machine: WAITING -> RUNNING (prefill -> decode) -> FINISHED."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

_id_counter = itertools.count(1)


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class FinishReason(str, Enum):
    STOP_TOKEN = "stop"
    MAX_TOKENS = "length"
    ABORTED = "aborted"


@dataclass
class Request:
    """
    A single generation request as it moves through the engine.

    prompt_token_ids are known up front. output_token_ids grows one at a time
    during decode, or in chunks during prefill continuation. num_computed_tokens
    tracks how many of the prompt tokens have actually been pushed through the
    (mock) model so far -- this is what makes chunked prefill possible: a
    request can sit "partially prefilled" across multiple scheduler steps.
    """

    request_id: str
    prompt_token_ids: List[int]
    max_new_tokens: int
    stop_token_id: Optional[int] = None

    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0  # prompt tokens processed through the model so far
    finish_reason: Optional[FinishReason] = None

    created_at: float = field(default_factory=time.monotonic)
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None

    @staticmethod
    def new_id() -> str:
        return f"req-{next(_id_counter)}"

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_len(self) -> int:
        """Total tokens currently occupying the KV cache for this request."""
        return self.prompt_len + len(self.output_token_ids)

    @property
    def is_prefill_complete(self) -> bool:
        return self.num_computed_tokens >= self.prompt_len

    @property
    def remaining_prompt_tokens(self) -> int:
        return max(0, self.prompt_len - self.num_computed_tokens)

    def record_prefill_progress(self, num_tokens: int) -> None:
        self.num_computed_tokens = min(self.prompt_len, self.num_computed_tokens + num_tokens)

    def append_output_token(self, token_id: int) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.monotonic()
        self.output_token_ids.append(token_id)
        self.num_computed_tokens += 1

    def should_stop(self) -> bool:
        if len(self.output_token_ids) >= self.max_new_tokens:
            self.finish_reason = FinishReason.MAX_TOKENS
            return True
        if self.stop_token_id is not None and self.output_token_ids:
            if self.output_token_ids[-1] == self.stop_token_id:
                self.finish_reason = FinishReason.STOP_TOKEN
                return True
        return False

    def mark_finished(self, reason: FinishReason) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
        self.finished_at = time.monotonic()

    @property
    def time_to_first_token(self) -> Optional[float]:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.created_at

    @property
    def total_latency(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return self.finished_at - self.created_at
