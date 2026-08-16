"""
FastAPI front end for the engine.

A single background task drives the engine's step() loop continuously.
Every HTTP request just enqueues work into the shared engine and awaits an
asyncio.Event that the loop fires once that specific request finishes -- so
concurrent callers hitting /v1/generate at the same time actually get batched
together into the same scheduler steps, which is the whole point of this
project. There is no per-request worker process or thread; everything shares
one event loop and one engine instance, matching how a single-GPU inference
server multiplexes concurrent clients in practice.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict

from dotenv import load_dotenv

load_dotenv()  # loads a local .env file if present; no-op otherwise

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

from ..engine import LLMEngine
from ..real_engine import RealLLMEngine
from ..backends.ollama_backend import OllamaConnectionError
from ..request import Request
from ..tokenizer import display_text, encode
from .schemas import (
    GenerateRequest,
    GenerateResponse,
    RealGenerateRequest,
    RealGenerateResponse,
    RealStatsResponse,
    RequestStatusResponse,
    StatsResponse,
)

logger = logging.getLogger("mini_vllm")

ENGINE_CONFIG = dict(
    num_blocks=int(os.getenv("MINI_VLLM_NUM_BLOCKS", 2048)),
    block_size=int(os.getenv("MINI_VLLM_BLOCK_SIZE", 16)),
    max_num_seqs=int(os.getenv("MINI_VLLM_MAX_NUM_SEQS", 32)),
    max_num_batched_tokens=int(os.getenv("MINI_VLLM_MAX_BATCHED_TOKENS", 4096)),
    max_prefill_chunk_tokens=int(os.getenv("MINI_VLLM_PREFILL_CHUNK", 512)),
)

REAL_ENGINE_CONFIG = dict(
    max_num_seqs=int(os.getenv("MINI_VLLM_REAL_MAX_NUM_SEQS", 4)),
    model=os.getenv("MINI_VLLM_OLLAMA_MODEL", "phi3:mini"),
    host=os.getenv("MINI_VLLM_OLLAMA_HOST", "http://localhost:11434"),
)

_engine: LLMEngine
_real_engine: RealLLMEngine
_completion_events: Dict[str, asyncio.Event] = {}
_loop_task: asyncio.Task


async def _scheduler_loop() -> None:
    idle_sleep = 0.005
    while True:
        if _engine.has_unfinished_requests():
            result = _engine.step()
            for finished in result.finished_requests:
                event = _completion_events.get(finished.request_id)
                if event is not None:
                    event.set()
            await asyncio.sleep(0)  # yield to request handlers between steps
        else:
            await asyncio.sleep(idle_sleep)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _real_engine, _loop_task
    _engine = LLMEngine(**ENGINE_CONFIG)
    _real_engine = RealLLMEngine(**REAL_ENGINE_CONFIG)
    _loop_task = asyncio.create_task(_scheduler_loop())
    logger.info("mini-vllm engine started with config: %s", ENGINE_CONFIG)
    logger.info("real (Ollama) engine configured with: %s", REAL_ENGINE_CONFIG)
    yield
    _loop_task.cancel()


app = FastAPI(
    title="mini-vLLM",
    description="A PagedAttention-style KV cache and continuous batching scheduler simulator.",
    version="0.1.0",
    lifespan=lifespan,
)


def _resolve_prompt_token_ids(req: GenerateRequest) -> list[int]:
    if req.prompt_token_ids is not None:
        return req.prompt_token_ids
    if req.prompt is not None:
        return encode(req.prompt)
    raise HTTPException(status_code=422, detail="either prompt or prompt_token_ids is required")


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    format: str = Query(
        "json",
        pattern="^(json|text)$",
        description="'text' returns plain UTF-8 text (real newlines, no JSON escaping) "
        "instead of the JSON envelope -- handy for reading output directly in a terminal.",
    ),
) -> GenerateResponse:
    prompt_token_ids = _resolve_prompt_token_ids(req)
    if not prompt_token_ids:
        raise HTTPException(status_code=422, detail="prompt encoded to zero tokens")

    request_id = Request.new_id()
    event = asyncio.Event()
    _completion_events[request_id] = event

    _engine.add_request(
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=req.max_tokens,
        stop_token_id=req.stop_token_id,
        request_id=request_id,
    )

    await event.wait()
    _completion_events.pop(request_id, None)

    r = _engine.get_request(request_id)
    ttft_ms = r.time_to_first_token * 1000 if r.time_to_first_token is not None else None
    latency_ms = r.total_latency * 1000 if r.total_latency is not None else None
    tps = (
        len(r.output_token_ids) / r.total_latency
        if r.total_latency and r.total_latency > 0
        else None
    )

    if format == "text":
        header = (
            f"# finish_reason={r.finish_reason.value if r.finish_reason else 'unknown'} "
            f"ttft_ms={ttft_ms:.1f} latency_ms={latency_ms:.1f} tokens={len(r.output_token_ids)}\n\n"
        )
        return PlainTextResponse(header + display_text(r.output_token_ids))

    return GenerateResponse(
        request_id=request_id,
        prompt_len=r.prompt_len,
        output_token_ids=r.output_token_ids,
        generated_text=display_text(r.output_token_ids),
        finish_reason=r.finish_reason.value if r.finish_reason else "unknown",
        time_to_first_token_ms=ttft_ms,
        total_latency_ms=latency_ms,
        tokens_generated=len(r.output_token_ids),
        tokens_per_second=tps,
    )


@app.post("/v1/generate/stream")
async def generate_stream(req: GenerateRequest):
    prompt_token_ids = _resolve_prompt_token_ids(req)
    if not prompt_token_ids:
        raise HTTPException(status_code=422, detail="prompt encoded to zero tokens")

    request_id = Request.new_id()
    event = asyncio.Event()
    _completion_events[request_id] = event

    _engine.add_request(
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=req.max_tokens,
        stop_token_id=req.stop_token_id,
        request_id=request_id,
    )

    async def token_stream():
        last_sent = 0
        r = _engine.get_request(request_id)
        yield f"event: start\ndata: {json.dumps({'request_id': request_id})}\n\n"
        while not event.is_set():
            if len(r.output_token_ids) > last_sent:
                new_tokens = r.output_token_ids[last_sent:]
                last_sent = len(r.output_token_ids)
                for tok in new_tokens:
                    yield f"event: token\ndata: {tok}\n\n"
            await asyncio.sleep(0.005)
        if len(r.output_token_ids) > last_sent:
            for tok in r.output_token_ids[last_sent:]:
                yield f"event: token\ndata: {tok}\n\n"
        _completion_events.pop(request_id, None)
        reason = r.finish_reason.value if r.finish_reason else "unknown"
        ttft_ms = r.time_to_first_token * 1000 if r.time_to_first_token is not None else None
        latency_ms = r.total_latency * 1000 if r.total_latency is not None else None
        done_payload = json.dumps(
            {
                "finish_reason": reason,
                "tokens_generated": len(r.output_token_ids),
                "time_to_first_token_ms": ttft_ms,
                "total_latency_ms": latency_ms,
            }
        )
        yield f"event: done\ndata: {done_payload}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.get("/v1/requests/{request_id}", response_model=RequestStatusResponse)
async def get_request_status(request_id: str) -> RequestStatusResponse:
    try:
        r = _engine.get_request(request_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown request_id")

    return RequestStatusResponse(
        request_id=r.request_id,
        status=r.status.value,
        prompt_len=r.prompt_len,
        num_computed_tokens=r.num_computed_tokens,
        num_output_tokens=len(r.output_token_ids),
        finish_reason=r.finish_reason.value if r.finish_reason else None,
    )


@app.get("/v1/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    return StatsResponse(
        num_running=len(_engine.scheduler.running),
        num_waiting=len(_engine.scheduler.waiting),
        cache=_engine.cache_manager.usage(),
        engine_config=ENGINE_CONFIG,
    )


@app.post("/v1/generate/real", response_model=RealGenerateResponse)
async def generate_real(
    req: RealGenerateRequest,
    format: str = Query(
        "json",
        pattern="^(json|text)$",
        description="'text' returns plain UTF-8 text (real newlines, no JSON escaping) "
        "instead of the JSON envelope -- handy for reading output directly in a terminal.",
    ),
) -> RealGenerateResponse:
    """
    Real text generation via a locally running Ollama model (default
    phi3:mini). This does NOT go through the block manager or scheduler --
    see the module docstring in real_engine.py for why that's an honest
    boundary rather than a missing feature. What's real here: the concurrency
    limit (MINI_VLLM_REAL_MAX_NUM_SEQS), and the timing/text you get back.
    """
    try:
        result = await _real_engine.generate(req.prompt, max_tokens=req.max_tokens)
    except OllamaConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if format == "text":
        header = (
            f"# finish_reason={result.finish_reason} "
            f"ttft_ms={result.time_to_first_token_ms} "
            f"latency_ms={result.total_latency_ms:.1f} "
            f"tokens={result.tokens_generated}\n\n"
        )
        return PlainTextResponse(header + result.generated_text)

    return RealGenerateResponse(
        request_id=result.request_id,
        generated_text=result.generated_text,
        finish_reason=result.finish_reason,
        time_to_first_token_ms=result.time_to_first_token_ms,
        total_latency_ms=result.total_latency_ms,
        tokens_generated=result.tokens_generated,
        tokens_per_second=result.tokens_per_second,
    )


@app.post("/v1/generate/real/stream")
async def generate_real_stream(req: RealGenerateRequest):
    async def token_stream():
        try:
            async for evt in _real_engine.generate_stream(req.prompt, max_tokens=req.max_tokens):
                if evt.type == "token":
                    yield f"event: token\ndata: {evt.text}\n\n"
                else:  # "done"
                    done_payload = json.dumps(
                        {
                            "finish_reason": evt.finish_reason,
                            "tokens_generated": evt.tokens_generated,
                            "time_to_first_token_ms": evt.time_to_first_token_ms,
                            "total_latency_ms": evt.total_latency_ms,
                        }
                    )
                    yield f"event: done\ndata: {done_payload}\n\n"
        except OllamaConnectionError as e:
            yield f'event: error\ndata: {json.dumps({"detail": str(e)})}\n\n'

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.get("/v1/generate/real/stats", response_model=RealStatsResponse)
async def real_stats() -> RealStatsResponse:
    s = await _real_engine.stats()
    return RealStatsResponse(
        num_running=s["num_running"],
        num_waiting=s["num_waiting"],
        max_num_seqs=_real_engine.max_num_seqs,
        model=_real_engine.model,
    )


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "timestamp": time.time()}