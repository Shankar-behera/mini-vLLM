from mini_vllm.block_manager import KVCacheManager
from mini_vllm.request import Request, RequestStatus
from mini_vllm.scheduler import Scheduler


def make_request(prompt_len=8, max_new_tokens=5, request_id=None) -> Request:
    return Request(
        request_id=request_id or Request.new_id(),
        prompt_token_ids=list(range(prompt_len)),
        max_new_tokens=max_new_tokens,
    )


def test_single_request_full_prefill_in_one_step_when_it_fits():
    mgr = KVCacheManager(num_blocks=100, block_size=16)
    sched = Scheduler(mgr, max_num_seqs=8, max_num_batched_tokens=2048, max_prefill_chunk_tokens=512)
    req = make_request(prompt_len=10)
    sched.add_request(req)

    out = sched.schedule()
    assert len(out.scheduled) == 1
    unit = out.scheduled[0]
    assert unit.is_prefill
    assert unit.num_tokens == 10
    assert unit.completes_prefill
    assert req.status == RequestStatus.RUNNING


def test_long_prompt_is_chunked_across_multiple_steps():
    mgr = KVCacheManager(num_blocks=100, block_size=16)
    sched = Scheduler(mgr, max_num_seqs=8, max_num_batched_tokens=2048, max_prefill_chunk_tokens=100)
    req = make_request(prompt_len=250)
    sched.add_request(req)

    out1 = sched.schedule()
    assert out1.scheduled[0].num_tokens == 100
    assert not out1.scheduled[0].completes_prefill

    out2 = sched.schedule()
    assert out2.scheduled[0].num_tokens == 100
    assert not out2.scheduled[0].completes_prefill

    out3 = sched.schedule()
    assert out3.scheduled[0].num_tokens == 50
    assert out3.scheduled[0].completes_prefill
    assert req.is_prefill_complete


def test_decode_requests_get_priority_over_new_prefill_chunk_budget():
    """
    Regression test for head-of-line blocking: a long prompt arriving after a
    request is already decoding should never stall that decode -- decode gets
    served first and prefill only eats leftover budget.
    """
    mgr = KVCacheManager(num_blocks=100, block_size=16)
    sched = Scheduler(mgr, max_num_seqs=8, max_num_batched_tokens=50, max_prefill_chunk_tokens=50)

    decoding_req = make_request(prompt_len=5, max_new_tokens=10)
    sched.add_request(decoding_req)
    sched.schedule()  # prefill (5 tokens) - completes immediately, next step decodes
    decoding_req.append_output_token(999)  # simulate engine producing first token

    long_req = make_request(prompt_len=1000, max_new_tokens=5)
    sched.add_request(long_req)

    out = sched.schedule()
    decode_units = [u for u in out.scheduled if u.request.request_id == decoding_req.request_id]
    prefill_units = [u for u in out.scheduled if u.request.request_id == long_req.request_id]

    assert len(decode_units) == 1
    assert decode_units[0].num_tokens == 1
    assert not decode_units[0].is_prefill
    # decode consumed 1 of the 50 token budget, prefill only gets the other 49
    assert len(prefill_units) == 1
    assert prefill_units[0].num_tokens == 49


def test_admission_blocked_when_no_free_blocks():
    mgr = KVCacheManager(num_blocks=4, block_size=16)  # 64 tokens total capacity
    sched = Scheduler(mgr, max_num_seqs=8, max_num_batched_tokens=2048, max_prefill_chunk_tokens=512)

    big_req = make_request(prompt_len=64)
    sched.add_request(big_req)
    out1 = sched.schedule()
    assert out1.scheduled[0].num_tokens == 64
    assert mgr.num_free_blocks == 0

    blocked_req = make_request(prompt_len=16)
    sched.add_request(blocked_req)
    out2 = sched.schedule()
    assert len(out2.scheduled) == 0  # no room, stays waiting
    assert blocked_req.status == RequestStatus.WAITING
    assert blocked_req in sched.waiting


def test_max_num_seqs_limits_concurrent_running_requests():
    mgr = KVCacheManager(num_blocks=100, block_size=16)
    sched = Scheduler(mgr, max_num_seqs=2, max_num_batched_tokens=2048, max_prefill_chunk_tokens=512)
    for _ in range(5):
        sched.add_request(make_request(prompt_len=4))

    sched.schedule()
    assert len(sched.running) == 2
    assert len(sched.waiting) == 3


def test_preemption_frees_blocks_for_higher_priority_decode():
    """
    Force a cache so tight that an existing decode step can't get a new block
    without evicting another running request, and confirm eviction happens
    and the request goes back to the waiting queue for recompute.
    """
    mgr = KVCacheManager(num_blocks=2, block_size=4)  # 8 tokens total capacity
    sched = Scheduler(mgr, max_num_seqs=8, max_num_batched_tokens=2048, max_prefill_chunk_tokens=512)

    req_a = make_request(prompt_len=4, max_new_tokens=10, request_id="A")
    req_b = make_request(prompt_len=4, max_new_tokens=10, request_id="B")
    sched.add_request(req_a)
    sched.add_request(req_b)
    sched.schedule()  # both fully prefilled: 2 blocks used, 0 free
    req_a.append_output_token(1)
    req_b.append_output_token(2)
    assert mgr.num_free_blocks == 0

    # Both are now full blocks (4/4 tokens) -> next decode token for either
    # needs a brand new block, and none are free. B (admitted after A) should
    # be preempted, per invocation order in `running`, to make room.
    out = sched.schedule()
    assert "B" in out.preempted_request_ids
    assert req_b.status == RequestStatus.WAITING
    assert req_b in sched.waiting
    assert not mgr.has_request("B")


def test_prioritize_decode_false_lets_prefill_stall_decode():
    """
    With prioritize_decode=False and no chunk cap, a large arriving prompt
    can consume the entire token budget before decode gets a look in --
    this is the historical failure mode chunked prefill (plus decode
    priority) exists to fix.
    """
    mgr = KVCacheManager(num_blocks=1000, block_size=16)
    sched = Scheduler(
        mgr,
        max_num_seqs=8,
        max_num_batched_tokens=500,
        max_prefill_chunk_tokens=500,
        prioritize_decode=False,
    )

    decoding_req = make_request(prompt_len=5, max_new_tokens=10)
    sched.add_request(decoding_req)
    sched.schedule()
    decoding_req.append_output_token(999)

    long_req = make_request(prompt_len=500, max_new_tokens=5)
    sched.add_request(long_req)

    out = sched.schedule()
    decode_units = [u for u in out.scheduled if u.request.request_id == decoding_req.request_id]
    prefill_units = [u for u in out.scheduled if u.request.request_id == long_req.request_id]
    assert prefill_units[0].num_tokens == 500  # consumed the entire budget
    assert len(decode_units) == 0  # decode got nothing this step


def test_remove_finished_frees_cache():
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    sched = Scheduler(mgr, max_num_seqs=8, max_num_batched_tokens=2048, max_prefill_chunk_tokens=512)
    req = make_request(prompt_len=4)
    sched.add_request(req)
    sched.schedule()
    assert mgr.num_free_blocks == 9

    sched.remove_finished(req)
    assert mgr.num_free_blocks == 10
    assert req not in sched.running
