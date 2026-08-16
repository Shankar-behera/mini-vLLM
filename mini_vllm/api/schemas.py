from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: Optional[str] = Field(
        None, description="Raw text, encoded with the toy hash tokenizer."
    )
    prompt_token_ids: Optional[List[int]] = Field(
        None, description="Explicit token ids, bypassing the toy tokenizer."
    )
    max_tokens: int = Field(64, gt=0, le=4096)
    stop_token_id: Optional[int] = None


class GenerateResponse(BaseModel):
    request_id: str
    prompt_len: int
    output_token_ids: List[int]
    generated_text: str = Field(
        description="Placeholder display text ('tok_<id>' per token) -- there's no real "
        "vocabulary behind this simulator's tokenizer. See /v1/generate/real for actual "
        "model output."
    )
    finish_reason: str
    time_to_first_token_ms: Optional[float]
    total_latency_ms: Optional[float]
    tokens_generated: int
    tokens_per_second: Optional[float]


class RequestStatusResponse(BaseModel):
    request_id: str
    status: str
    prompt_len: int
    num_computed_tokens: int
    num_output_tokens: int
    finish_reason: Optional[str]


class StatsResponse(BaseModel):
    num_running: int
    num_waiting: int
    cache: dict
    engine_config: dict


class RealGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Raw text prompt sent directly to the real model.")
    max_tokens: int = Field(128, gt=0, le=4096)


class RealGenerateResponse(BaseModel):
    request_id: str
    generated_text: str
    finish_reason: str
    time_to_first_token_ms: Optional[float]
    total_latency_ms: float
    tokens_generated: int
    tokens_per_second: Optional[float]


class RealStatsResponse(BaseModel):
    num_running: int
    num_waiting: int
    max_num_seqs: int
    model: str
