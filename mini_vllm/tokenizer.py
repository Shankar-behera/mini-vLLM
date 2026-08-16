"""
A deterministic, dependency-free stand-in tokenizer.

mini-vLLM simulates the memory management and scheduling machinery of an
inference server, not language modeling itself, so there's no real vocabulary
or detokenizer behind this. Text in gets turned into stable integer ids via a
hash, which is all the rest of the system needs to exercise real block
allocation and scheduling behavior against realistic-looking sequence
lengths. Output "tokens" are reported as ids, not reconstructed text.
"""

from __future__ import annotations

import hashlib
from typing import List


def encode(text: str, vocab_size: int = 32000) -> List[int]:
    words = text.strip().split()
    if not words:
        return []
    ids = []
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        token_id = int.from_bytes(digest[:4], "big") % vocab_size
        ids.append(token_id)
    return ids


def display_text(token_ids: List[int]) -> str:
    """Placeholder text for token ids with no real vocabulary behind them."""
    return " ".join(f"tok_{t}" for t in token_ids)
