import os

from mini_vllm.block_manager import KVCacheManager
from mini_vllm.engine import LLMEngine
from mini_vllm.visualize import (
    render_blocks_ascii,
    render_blocks_png,
    render_running_waiting_ascii,
    render_step_ascii,
)


def test_render_blocks_ascii_shows_correct_counts():
    mgr = KVCacheManager(num_blocks=16, block_size=4)
    mgr.allocate("req-1", num_tokens=8)  # 2 blocks
    output = render_blocks_ascii(mgr, width=8)
    assert "2/16" in output
    assert output.count("■") == 2
    assert output.count("□") == 14


def test_render_running_waiting_ascii_lists_ids():
    engine = LLMEngine(num_blocks=64, block_size=16, max_num_seqs=1)
    engine.add_request(prompt_token_ids=list(range(5)), max_new_tokens=3, request_id="A")
    engine.add_request(prompt_token_ids=list(range(5)), max_new_tokens=3, request_id="B")
    engine.step()  # only A should be admitted, max_num_seqs=1

    output = render_running_waiting_ascii(engine.scheduler)
    assert "running (1): A" in output
    assert "waiting (1): B" in output


def test_render_step_ascii_combines_both_views():
    engine = LLMEngine(num_blocks=64, block_size=16)
    engine.add_request(prompt_token_ids=list(range(5)), max_new_tokens=3, request_id="A")
    engine.step()
    output = render_step_ascii(engine.cache_manager, engine.scheduler, step_num=1)
    assert "Iteration 1" in output
    assert "running" in output
    assert "blocks:" in output


def test_render_blocks_png_writes_a_file(tmp_path):
    mgr = KVCacheManager(num_blocks=20, block_size=4)
    mgr.allocate("req-1", num_tokens=8)
    out_path = tmp_path / "blocks.png"
    render_blocks_png(mgr, str(out_path), width=5)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
