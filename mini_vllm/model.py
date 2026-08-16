"""
A stand-in "model" so the engine has something to actually execute.

This does not implement a language model. It runs a real (tiny) matmul sized
by batch tokens and hidden_dim so that batching more tokens together costs
proportionally more wall-clock time, same as a real transformer forward pass,
and produces token ids via argmax over a random projection. That's enough to
let the scheduler and engine be benchmarked under realistic-shaped latency
without pulling in an actual LLM.
"""

from __future__ import annotations

import numpy as np


class MockModel:
    def __init__(self, hidden_dim: int = 256, vocab_size: int = 32000, seed: int = 0):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        # fixed random projection stands in for the model's output head
        self._lm_head = rng.standard_normal((hidden_dim, vocab_size)).astype(np.float32)

    def forward_batch(self, num_tokens: int) -> np.ndarray:
        """
        Simulates one forward pass over num_tokens (summed across every
        request scheduled this step). Returns hidden states of shape
        (num_tokens, hidden_dim). The actual matmuls here are what give the
        benchmark believable, batch-size-dependent latency.
        """
        if num_tokens <= 0:
            return np.zeros((0, self.hidden_dim), dtype=np.float32)

        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        x = rng.standard_normal((num_tokens, self.hidden_dim)).astype(np.float32)
        w1 = rng.standard_normal((self.hidden_dim, self.hidden_dim)).astype(np.float32)
        w2 = rng.standard_normal((self.hidden_dim, self.hidden_dim)).astype(np.float32)

        h = np.tanh(x @ w1)
        h = np.tanh(h @ w2)
        return h

    def sample_next_tokens(self, hidden_states: np.ndarray) -> np.ndarray:
        """Greedy-decode a token id per row of hidden_states."""
        if hidden_states.shape[0] == 0:
            return np.array([], dtype=np.int64)
        logits = hidden_states @ self._lm_head
        return np.argmax(logits, axis=-1)
