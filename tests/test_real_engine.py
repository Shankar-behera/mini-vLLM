import asyncio

import pytest

from mini_vllm.real_engine import RealLLMEngine


class _FakeBackend:
    """Drop-in replacement for OllamaBackend that yields controlled chunks."""

    def __init__(self, pieces, delay=0.0):
        self.pieces = pieces
        self.delay = delay
        self.call_count = 0

    async def astream(self, prompt, model, max_tokens, client=None):
        self.call_count += 1
        for i, piece in enumerate(self.pieces):
            if self.delay:
                await asyncio.sleep(self.delay)
            done = i == len(self.pieces) - 1
            yield {"response": piece, "done": done, "done_reason": "stop" if done else None}


@pytest.mark.asyncio
async def test_generate_aggregates_streamed_text():
    engine = RealLLMEngine(max_num_seqs=4, model="phi3:mini")
    engine.backend = _FakeBackend(["Paged", "Attention", " rocks"])

    result = await engine.generate("explain kv caching", max_tokens=10)

    assert result.generated_text == "PagedAttention rocks"
    assert result.finish_reason == "stop"
    assert result.tokens_generated == 3
    assert result.time_to_first_token_ms is not None
    assert result.total_latency_ms >= 0


@pytest.mark.asyncio
async def test_concurrency_limit_actually_queues_excess_requests():
    """
    Fire more concurrent requests than max_num_seqs allows and confirm some
    of them genuinely wait (num_waiting > 0 at some point) rather than all
    being handed to the backend at once.
    """
    engine = RealLLMEngine(max_num_seqs=2, model="phi3:mini")
    engine.backend = _FakeBackend(["a", "b"], delay=0.05)

    async def track_max_waiting():
        max_waiting = 0
        for _ in range(20):
            s = await engine.stats()
            max_waiting = max(max_waiting, s["num_waiting"])
            await asyncio.sleep(0.01)
        return max_waiting

    tasks = [asyncio.create_task(engine.generate("hi", max_tokens=5)) for _ in range(6)]
    watcher = asyncio.create_task(track_max_waiting())

    results = await asyncio.gather(*tasks)
    max_waiting_seen = await watcher

    assert len(results) == 6
    assert all(r.generated_text == "ab" for r in results)
    assert max_waiting_seen > 0  # proves admission control actually queued someone


@pytest.mark.asyncio
async def test_stats_returns_to_zero_after_all_requests_finish():
    engine = RealLLMEngine(max_num_seqs=3, model="phi3:mini")
    engine.backend = _FakeBackend(["done"])

    await asyncio.gather(*[engine.generate("hi", max_tokens=5) for _ in range(5)])

    final_stats = await engine.stats()
    assert final_stats == {"num_running": 0, "num_waiting": 0}


@pytest.mark.asyncio
async def test_generate_stream_yields_token_events_then_done():
    engine = RealLLMEngine(max_num_seqs=2, model="phi3:mini")
    engine.backend = _FakeBackend(["one", "two", "three"])

    events = [e async for e in engine.generate_stream("hi", max_tokens=5)]

    token_events = [e for e in events if e.type == "token"]
    done_events = [e for e in events if e.type == "done"]

    assert [e.text for e in token_events] == ["one", "two", "three"]
    assert len(done_events) == 1
    assert done_events[0].finish_reason == "stop"
    assert done_events[0].tokens_generated == 3
    assert done_events[0].time_to_first_token_ms is not None
    assert done_events[0].total_latency_ms is not None


@pytest.mark.asyncio
async def test_generate_stream_done_event_reports_length_when_truncated():
    """
    Regression test: previously the streaming path gave no way to tell
    whether generation stopped naturally or got cut off mid-sentence by
    max_tokens -- the done event must carry the real done_reason through.
    """
    engine = RealLLMEngine(max_num_seqs=2, model="phi3:mini")

    class TruncatedBackend:
        async def astream(self, prompt, model, max_tokens, client=None):
            yield {"response": "partial", "done": True, "done_reason": "length"}

    engine.backend = TruncatedBackend()

    events = [e async for e in engine.generate_stream("hi", max_tokens=1)]
    done_event = next(e for e in events if e.type == "done")
    assert done_event.finish_reason == "length"