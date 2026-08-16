from mini_vllm.benchmark import benchmark_chunked_vs_monolithic_prefill, benchmark_naive_vs_continuous_batching


def test_continuous_batching_beats_naive_serial_throughput():
    naive, continuous = benchmark_naive_vs_continuous_batching(num_requests=20)
    assert continuous.throughput_tokens_per_s > naive.throughput_tokens_per_s
    assert continuous.wall_clock_s < naive.wall_clock_s


def test_chunked_prefill_reduces_short_request_ttft():
    unbounded, chunked = benchmark_chunked_vs_monolithic_prefill()
    # the whole point of chunking: short, already-waiting requests get their
    # first token much sooner when a huge prompt can't monopolize every step
    assert chunked.short_requests_mean_ttft_ms < unbounded.short_requests_mean_ttft_ms
    # the trade-off: the long request itself takes longer to finish
    assert chunked.long_request_latency_ms > unbounded.long_request_latency_ms
