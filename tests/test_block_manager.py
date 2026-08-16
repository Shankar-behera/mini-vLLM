import pytest

from mini_vllm.block_manager import KVCacheManager, OutOfMemoryError


def test_allocate_rounds_up_to_block_boundary():
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    table = mgr.allocate("req-1", num_tokens=5)
    assert len(table.physical_blocks) == 2  # 5 tokens -> ceil(5/4) = 2 blocks
    assert mgr.num_free_blocks == 8


def test_allocate_exact_block_boundary():
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    table = mgr.allocate("req-1", num_tokens=8)
    assert len(table.physical_blocks) == 2
    assert table.is_last_block_full()


def test_allocate_raises_when_out_of_memory():
    mgr = KVCacheManager(num_blocks=2, block_size=4)
    with pytest.raises(OutOfMemoryError):
        mgr.allocate("req-1", num_tokens=100)


def test_double_allocate_same_request_raises():
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    mgr.allocate("req-1", num_tokens=4)
    with pytest.raises(ValueError):
        mgr.allocate("req-1", num_tokens=1)


def test_append_token_reuses_partial_block():
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    mgr.allocate("req-1", num_tokens=1)  # 1 block, 3 free slots left in it
    free_before = mgr.num_free_blocks
    new_block = mgr.append_token("req-1")
    assert new_block is None  # fit in existing block, no new physical block used
    assert mgr.num_free_blocks == free_before


def test_append_token_grabs_new_block_when_full():
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    mgr.allocate("req-1", num_tokens=4)  # exactly fills one block
    free_before = mgr.num_free_blocks
    new_block = mgr.append_token("req-1")
    assert new_block is not None
    assert mgr.num_free_blocks == free_before - 1


def test_append_chunk_matches_repeated_append_token():
    mgr_a = KVCacheManager(num_blocks=20, block_size=4)
    mgr_a.allocate("req-1", num_tokens=3)
    mgr_a.append_chunk("req-1", 10)

    mgr_b = KVCacheManager(num_blocks=20, block_size=4)
    mgr_b.allocate("req-1", num_tokens=3)
    for _ in range(10):
        mgr_b.append_token("req-1")

    table_a = mgr_a.get_block_table("req-1")
    table_b = mgr_b.get_block_table("req-1")
    assert table_a.num_tokens == table_b.num_tokens
    assert len(table_a.physical_blocks) == len(table_b.physical_blocks)


def test_free_returns_blocks_to_pool_no_fragmentation():
    mgr = KVCacheManager(num_blocks=4, block_size=4)
    mgr.allocate("req-1", num_tokens=8)  # 2 of the 4 blocks
    mgr.free("req-1")
    assert mgr.num_free_blocks == 4  # fully reclaimed, no leaked/fragmented blocks


def test_free_unknown_request_is_a_noop():
    mgr = KVCacheManager(num_blocks=4, block_size=4)
    mgr.free("does-not-exist")  # should not raise


def test_no_fragmentation_across_alloc_free_cycles():
    """
    Repeatedly allocate/free requests of varying odd sizes and confirm the
    pool always returns to full capacity -- this is the actual guarantee
    block-based allocation is supposed to give you over a naive contiguous
    buffer per request.
    """
    mgr = KVCacheManager(num_blocks=16, block_size=4)
    for cycle in range(50):
        ids = [f"r{cycle}-{i}" for i in range(3)]
        for i, rid in enumerate(ids):
            mgr.allocate(rid, num_tokens=(i + 1) * 3)  # 3, 6, 9 tokens
        for rid in ids:
            mgr.free(rid)
    assert mgr.num_free_blocks == 16


def test_usage_reports_correct_utilization():
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    mgr.allocate("req-1", num_tokens=8)  # 2 blocks used
    usage = mgr.usage()
    assert usage["used_blocks"] == 2
    assert usage["free_blocks"] == 8
    assert usage["utilization_pct"] == 20.0
