# mini-vLLM

A from-scratch simulator of the two ideas that make modern LLM inference servers
(vLLM, TensorRT-LLM, etc.) actually work at scale:

1. **PagedAttention-style KV cache management** — fixed-size memory blocks
   allocated on demand and mapped through a per-request block table, instead
   of reserving a worst-case contiguous buffer per sequence.
2. **Continuous batching with chunked prefill** — an iteration-level
   scheduler that admits, runs, and evicts requests one *token* at a time
   instead of one *batch* at a time, and splits long prompts into bounded
   chunks so they can't stall requests that are already generating.

It ships as an installable Python package, a FastAPI service you can run
locally or in Docker, a benchmark suite that puts numbers behind both
claims, visualizations of the scheduler and cache in action, and an optional
real-generation mode that streams actual text from a local Ollama model.

## What this is not

There's no real language model behind the core simulator. The "model" used
by the scheduler/benchmarks is a small stack of `numpy` matmuls that
produces plausible-shaped compute cost and greedy-decodes token ids from
random logits (see `mini_vllm/model.py`). That's deliberate — the goal is to
reproduce the *memory management and scheduling* problem real inference
engines solve, which is orthogonal to what the actual transformer weights
compute. If you plug in a real model's forward pass in place of `MockModel`,
the block manager and scheduler don't need to change.

That said, this isn't limited to the mock model — see **Real generation
mode** below for wiring an actual local model (e.g. `phi3:mini` via Ollama)
in as a second, honestly-scoped path.

## Architecture

```
┌─────────────┐     add_request()      ┌────────────────┐
│   FastAPI    │ ─────────────────────▶ │   LLMEngine     │
│  (api/app.py)│ ◀───────────────────── │  (engine.py)    │
└─────────────┘   generated tokens      └───────┬────────┘
                                                 │ step()
                        ┌────────────────────────┼────────────────────────┐
                        │                        │                        │
                        ▼                        ▼                        ▼
              ┌──────────────────┐    ┌────────────────────┐   ┌──────────────────┐
              │    Scheduler      │    │   KVCacheManager    │   │    Model      │
              │  (scheduler.py)   │◀──▶│  (block_manager.py) │   │   (model.py)      │
              │                    │    │                     │   │                   │
              │ - waiting queue    │    │ - free block pool   │   │ - forward_batch() │
              │ - running list     │    │ - block tables      │   │ - sample tokens   │
              │ - admission control│    │ - alloc/append/free │   │                   │
              │ - chunked prefill  │    └────────────────────┘   └──────────────────┘
              │ - preemption       │
              └────────────────────┘
```

Every call to `engine.step()` is one "iteration" of the server:

1. `Scheduler.schedule()` decides what work happens this iteration — which
   already-running requests get a decode token, which in-flight prefills get
   another chunk, and which waiting requests get admitted — while
   `KVCacheManager` enforces that none of it happens without the physical
   blocks to back it.
2. `MockModel.forward_batch()` runs a real (tiny) forward pass sized by the
   iteration's total token count.
3. Any request whose generation just finished gets its blocks freed
   immediately, and the freed blocks are available to the very next
   iteration — no waiting for a batch boundary.

### Why block-based allocation matters

A naive implementation reserves `max_sequence_length` worth of KV cache for
every request the moment it arrives, whether or not it ever generates that
many tokens. That's the actual cause of the VRAM fragmentation and low
utilization that motivated PagedAttention. Here, `KVCacheManager` hands out
fixed-size blocks only as a sequence grows into them (`tests/test_block_manager.py`
has a fragmentation-cycle test that allocates and frees varying-size requests
50 times over and confirms the pool always returns to full capacity).

### Why chunked prefill matters

Without it, a newly-arrived long prompt either has to wait behind everything
else in a FIFO queue, or — if admitted — can consume an entire iteration's
compute budget by itself, during which nothing else makes progress. Both are
forms of head-of-line blocking. `Scheduler` fixes this two ways:

- Decode requests always get first claim on each iteration's token budget,
  so a sequence that's already generating is never skipped in favor of a new
  prompt (`test_decode_requests_get_priority_over_new_prefill_chunk_budget`).
- Prefill work — for both in-flight and newly-admitted requests — is capped
  per iteration by `max_prefill_chunk_tokens`, so even when there's budget
  left over after decode, one huge prompt can't eat all of it and starve
  other requests still sitting in the FIFO queue.

The benchmark section below has actual numbers for what this buys you.

## Seeing it happen

`scripts/visualize_demo.py` runs a small, deliberately visualization-friendly
workload (5 requests, `max_num_seqs=3` so two of them have to wait) and
renders both an ASCII trace and two PNGs.

ASCII, per iteration (`render_step_ascii`) — request C's long prompt is
still being chunked-prefilled in iteration 1 while A and B are already
admitted, and blocks are visibly returned to the free pool the moment a
request finishes:

```
--------------------------------------------------
Iteration 6
--------------------------------------------------
running (2): A, B
waiting (2): D, E

blocks: 5/40 used (12.5%)
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□■□□■■■
```

The same run as a timeline (`benchmarks/schedule_timeline.png`) — each row
is a request, each column an iteration, color shows prefill vs. decode vs.
waiting vs. finished. You can watch D and E sit in `waiting` until A or B
frees a slot in the running set:

![scheduling timeline](benchmarks/schedule_timeline.png)

And a snapshot of the block pool itself (`benchmarks/block_allocation.png`)
partway through the same run:

![block allocation](benchmarks/block_allocation.png)

```bash
python scripts/visualize_demo.py
```


## Project layout

```
mini_vllm/
  block_manager.py   KV cache block allocator
  request.py          request lifecycle / state machine
  scheduler.py        iteration-level scheduler, chunked prefill, preemption
  model.py             mock model (real matmuls, fake weights)
  engine.py            wires cache + scheduler + model into a step loop
  tokenizer.py         deterministic hash tokenizer (see note above)
  benchmark.py         benchmark scenarios + stats
  visualize.py         ASCII + PNG renderings of scheduler/cache state
  real_engine.py        concurrency-controlled engine for real generation
  backends/
    ollama_backend.py  async client for a local Ollama server
  api/
    app.py             FastAPI app, background scheduler loop, SSE streaming
    schemas.py         pydantic request/response models
scripts/
  run_benchmark.py     CLI entry point, prints tables + saves a chart
  visualize_demo.py    runs a small workload, saves timeline + block PNGs
tests/                 49 tests covering all of the above
benchmarks/            results.png, schedule_timeline.png, block_allocation.png
.env.example           template for all engine + real-generation-mode config
```

## Install

Requires Python 3.10+.

```bash
git clone <your-fork-url>
cd mini-vllm
pip install -e ".[dev]"        # editable install + test/benchmark deps
```

or, without the package/pyproject machinery:

```bash
pip install -r requirements-dev.txt
```

## Run the tests

```bash
pytest -v
```

49 tests, covering block allocation and fragmentation, scheduler admission
control and preemption, end-to-end engine runs, the FastAPI layer (including
a real concurrency test with 8 simultaneous requests), both benchmark
claims as regression tests, the visualization helpers, and the Ollama
backend / real-generation engine (against a mocked HTTP transport, so no
Ollama installation is required to run the suite).

## Run the benchmarks

```bash
python scripts/run_benchmark.py --requests 40
```

This runs two scenarios and writes `benchmarks/results.png`.

**Scenario A — naive serial processing vs. continuous batching.** 40
requests with randomized prompt/generation lengths, submitted all at once.
`max_num_seqs=1` forces the engine to fully finish one request before
starting the next (what you'd get from a plain request queue in front of a
single-sequence generation loop); `max_num_seqs=16` lets the scheduler batch
them together across iterations. Representative numbers from one run:

| scenario | wall clock | throughput | p50 latency | p99 latency |
|---|---|---|---|---|
| naive (max_num_seqs=1) | 6.4s | 287 tok/s | 3099 ms | 6443 ms |
| continuous batching (max_num_seqs=16) | 1.6s | 1133 tok/s | 952 ms | 1632 ms |

Same total work, ~4x the throughput, ~4x lower latency, purely from letting
independent requests share iterations instead of running strictly one after
another.

*Independently reproduced on different hardware:* a second run (different
machine) came out even more pronounced — 170 → 1220 tok/s, a ~7x gain — same
direction, bigger gap. Absolute numbers are wall-clock and will move between
machines; the effect itself doesn't.

**Scenario B — chunked prefill vs. head-of-line blocking.** One 4000-token
prompt mixed with 15 short, decode-heavy requests, all submitted together.
Both runs use the same overall per-iteration token budget (512); the only
difference is whether a single request's prefill chunk is capped well below
that budget (64) or allowed to claim the whole thing. This is reported
per-request-class rather than aggregated, because the aggregate numbers
average away the actual effect:

| scenario | long request TTFT | long request latency | short requests mean TTFT |
|---|---|---|---|
| unbounded prefill chunk | 58 ms | 117 ms | 73 ms |
| chunked prefill (cap=64) | 598 ms | 612 ms | 21 ms |

Capping the chunk size cuts the short requests' time-to-first-token by
roughly 3.5x — because the long prompt can no longer eat the entire FIFO
queue's turn — at the direct cost of the long request itself taking about
5x longer to finish. That trade-off is the actual point: chunked prefill
isn't free, it's a latency-fairness knob, and `max_prefill_chunk_tokens` is
how you tune where on that curve you sit.

(Run it yourself — these numbers move a bit between runs since the "model"
cost is real wall-clock matmul time on whatever CPU you're running on.
Same caveat as above: reproduced on a second machine with different
absolute numbers — 120ms → 16ms short-request TTFT, ~7.5x — but the same
direction and the same trade-off against the long request's own latency.)

## Run the API locally

```bash
uvicorn mini_vllm.api.app:app --reload
```

Then:

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "the quick brown fox jumps", "max_tokens": 20}'
```

```json
{
  "request_id": "req-1",
  "prompt_len": 5,
  "output_token_ids": [29560, 8334, 26275, ...],
  "generated_text": "tok_29560 tok_8334 tok_26275 ...",
  "finish_reason": "length",
  "time_to_first_token_ms": 32.8,
  "total_latency_ms": 61.5,
  "tokens_generated": 20,
  "tokens_per_second": 97.6
}
```

Streaming (server-sent events, one `token` event per generated token):

```bash
curl -N -X POST http://localhost:8000/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "stream this", "max_tokens": 10}'
```

Other endpoints: `GET /v1/requests/{request_id}` for polling a request's
status, `GET /v1/stats` for live cache utilization and running/waiting
counts, `GET /healthz` for liveness.

Interactive API docs are auto-generated by FastAPI at `/docs`.

Fire several `/v1/generate` calls at once (e.g. from separate terminals or
a small load-testing script) and hit `/v1/stats` while they're in flight —
you'll see `num_running` go above 1 and the requests all complete around the
same time, which is continuous batching actually happening, not just a
config flag.

### Configuration

All of it is environment variables, read at startup:

| variable | default | meaning |
|---|---|---|
| `MINI_VLLM_NUM_BLOCKS` | 2048 | total physical KV cache blocks |
| `MINI_VLLM_BLOCK_SIZE` | 16 | tokens per block |
| `MINI_VLLM_MAX_NUM_SEQS` | 32 | max concurrently running requests |
| `MINI_VLLM_MAX_BATCHED_TOKENS` | 4096 | total token budget per iteration |
| `MINI_VLLM_PREFILL_CHUNK` | 512 | max prefill tokens per request per iteration |

A template with all of these (plus the real-generation-mode variables below)
is in `.env.example` — copy it to `.env`, adjust, and it's picked up
automatically (`app.py` calls `load_dotenv()` on startup, no manual export
needed). Docker Compose users: pass it through `env_file:` instead, since
`.env` isn't copied into the image.

## Real generation mode (optional, via Ollama)

Everything above uses the mock model so the scheduler's behavior is
measurable and reproducible. If you want to see actual generated text
instead of token ids, there's a second path that talks to a real model
running locally through [Ollama](https://ollama.com):

```bash
ollama serve                # in one terminal
ollama pull phi3:mini       # once
```

Then, with the FastAPI service running:

```bash
curl -X POST http://localhost:8000/v1/generate/real \
  -H "Content-Type: application/json" \
  -d '{"prompt": "In one sentence, what is KV cache paging?", "max_tokens": 60}'
```

```json
{
  "request_id": "req-3",
  "generated_text": "KV cache paging splits a sequence's key/value cache into fixed-size blocks so memory can be allocated on demand instead of reserved upfront...",
  "finish_reason": "stop",
  "time_to_first_token_ms": 210.4,
  "total_latency_ms": 1840.2,
  "tokens_generated": 42,
  "tokens_per_second": 22.8
}
```

JSON always escapes newlines as `\n` — correct per spec, but unreadable if
you're just eyeballing curl output, especially for longer answers with
code blocks or multiple paragraphs. Add `?format=text` to get plain UTF-8
text back instead (real newlines, no JSON envelope, a one-line header with
the stats):

```bash
curl -X POST "http://localhost:8000/v1/generate/real?format=text" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what a Python dictionary is.", "max_tokens": 512}'
```

```
# finish_reason=stop ttft_ms=8901.2 latency_ms=25528.3 tokens=364

A Python dictionary, also known as an associative array or hash map in
other programming languages, is a collection of key-value pairs...
```

Same `?format=text` option works on `/v1/generate` (the mock endpoint) too.
Note this doesn't render markdown — a terminal can't do that regardless of
what the API returns — it just gets you real line breaks instead of `\n`.

Streaming version: `POST /v1/generate/real/stream` (SSE, one `token` event
per generated piece, followed by a final `done` event carrying
`finish_reason` — `"stop"` for a natural end, `"length"` if it got cut off
by `max_tokens` — plus `tokens_generated`, `time_to_first_token_ms`, and
`total_latency_ms`). `GET /v1/generate/real/stats` reports how many real
requests are currently running vs. queued.

**What's real here and what isn't, precisely:** the model, the text, and
the timing are all genuine. The concurrency limit
(`MINI_VLLM_REAL_MAX_NUM_SEQS`, default 4) is a real semaphore — send more
concurrent requests than that and the extras genuinely queue before Ollama
ever sees them, which `tests/test_real_engine.py` verifies directly. What
this path does *not* do is run through `Scheduler` or `KVCacheManager` —
Ollama's HTTP API is a black-box completion endpoint with no hook for
interleaving multiple sequences into one forward pass or exposing its KV
cache for block-level management, so there's nothing honest to wire up
there. The block-based paging and iteration-level scheduling this project
demonstrates are fully real against the mock model; layering them on top of
an opaque model server would just be decoration. See
`mini_vllm/real_engine.py`'s docstring for the longer version.

Configuration: `MINI_VLLM_OLLAMA_MODEL` (default `phi3:mini`),
`MINI_VLLM_OLLAMA_HOST` (default `http://localhost:11434`),
`MINI_VLLM_REAL_MAX_NUM_SEQS` (default 4).

If Ollama isn't running, `/v1/generate/real` returns a `503` with a message
telling you to start it — see `tests/test_api.py::test_generate_real_returns_503_when_ollama_unreachable`.

## Deploy with Docker

```bash
docker build -t mini-vllm .
docker run -p 8000:8000 mini-vllm
```

or

```bash
docker compose up --build
```

The image installs the package from `pyproject.toml`, exposes port 8000,
and has a `HEALTHCHECK` against `/healthz`.

## Known simplifications

Documented here rather than left implicit:

- **No real tokenizer.** `tokenizer.py` hashes whitespace-split words into
  token ids deterministically. Output is reported as token ids, not
  reconstructed text. Swapping in a real tokenizer wouldn't touch anything
  in `block_manager.py`, `scheduler.py`, or `engine.py`.
- **Schedule and execute are one step.** Real engines split "decide what to
  run" from "run it" because a real GPU forward pass can fail (OOM) after
  the decision is made. Here the mock model can't fail, so `Scheduler.schedule()`
  commits its allocation decisions immediately instead of staging them.
- **Preemption is simplified recompute.** When the cache is full, the
  evicted request's KV progress is discarded and it goes back to the front
  of the waiting queue with `num_computed_tokens` reset to 0. A fully
  faithful recompute policy would fold already-generated output tokens back
  into the context being re-prefilled; this skips that for simplicity.
- **No prefix caching / block sharing.** Every request gets its own private
  blocks. Real PagedAttention implementations also support sharing blocks
  across requests with identical prompt prefixes (e.g. a shared system
  prompt) via reference counting and copy-on-write — a natural extension if
  you want to take this further.
- **Single process, single event loop.** The FastAPI service runs one engine
  instance in one process. There's no multi-GPU, tensor-parallel, or
  multi-worker story here — this is a scheduling and memory-management
  simulator, not a distributed systems one.

## License

MIT — see `LICENSE`.