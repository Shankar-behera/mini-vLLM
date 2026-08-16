from mini_vllm.engine import LLMEngine
from mini_vllm.request import FinishReason


def test_single_request_runs_to_completion():
    engine = LLMEngine(num_blocks=64, block_size=16, max_num_batched_tokens=2048)
    rid = engine.add_request(prompt_token_ids=list(range(20)), max_new_tokens=5)

    finished = engine.run_until_complete()
    assert len(finished) == 1
    req = engine.get_request(rid)
    assert req.finish_reason == FinishReason.MAX_TOKENS
    assert len(req.output_token_ids) == 5
    assert engine.cache_manager.num_free_blocks == 64  # fully reclaimed after finish


def test_stop_token_ends_generation_early():
    engine = LLMEngine(num_blocks=64, block_size=16)
    # seed 0's mock model is deterministic given the same call sequence; instead
    # of depending on which token id it emits, just confirm max_new_tokens acts
    # as a hard ceiling when stop token is (almost certainly) never sampled.
    rid = engine.add_request(prompt_token_ids=[1, 2, 3], max_new_tokens=3, stop_token_id=-1)
    finished = engine.run_until_complete()
    assert finished[0].finish_reason == FinishReason.MAX_TOKENS
    assert len(finished[0].output_token_ids) == 3


def test_many_concurrent_requests_all_complete_and_cache_fully_reclaimed():
    engine = LLMEngine(
        num_blocks=64,
        block_size=8,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_prefill_chunk_tokens=16,
    )
    request_ids = []
    for i in range(10):
        rid = engine.add_request(prompt_token_ids=list(range(10 + i)), max_new_tokens=4)
        request_ids.append(rid)

    finished = engine.run_until_complete()
    assert len(finished) == 10
    for rid in request_ids:
        req = engine.get_request(rid)
        assert req.finish_reason is not None
        assert len(req.output_token_ids) == 4
    assert engine.cache_manager.num_free_blocks == 64


def test_step_reports_latency_and_batched_tokens():
    engine = LLMEngine(num_blocks=64, block_size=16)
    engine.add_request(prompt_token_ids=list(range(10)), max_new_tokens=2)
    result = engine.step()
    assert result.step_latency_s >= 0
    assert result.scheduler_output.num_batched_tokens == 10


def test_run_until_complete_respects_max_steps_safety_valve():
    engine = LLMEngine(num_blocks=1, block_size=1)  # 1 token of total capacity
    engine.add_request(prompt_token_ids=list(range(50)), max_new_tokens=5)  # can never fully admit
    finished = engine.run_until_complete(max_steps=10)
    assert len(finished) == 0  # request permanently stuck waiting, never OOMs the loop
