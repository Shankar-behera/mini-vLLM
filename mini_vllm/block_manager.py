"""
Block-based KV cache allocator.

Physical KV cache memory is carved into fixed-size blocks up front (like pages
in an OS). Each request gets a logical block table that maps its sequence
position to a physical block id. Blocks are only handed out as a sequence
actually needs them, and freed back to a shared pool the moment a request
finishes, so no request ever pre-reserves memory for tokens it hasn't
generated yet. That's the core trick vLLM uses to eliminate KV cache
fragmentation, reproduced here at a toy scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


class OutOfMemoryError(RuntimeError):
    """Raised when the cache manager cannot satisfy an allocation request."""


@dataclass
class BlockTable:
    """Logical-to-physical block mapping for a single request."""

    request_id: str
    block_size: int
    physical_blocks: List[int] = field(default_factory=list)
    num_tokens: int = 0

    def num_logical_blocks(self) -> int:
        if self.num_tokens == 0:
            return 0
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def slots_in_last_block(self) -> int:
        if self.num_tokens == 0:
            return 0
        remainder = self.num_tokens % self.block_size
        return self.block_size if remainder == 0 else remainder

    def is_last_block_full(self) -> bool:
        return self.num_tokens > 0 and self.num_tokens % self.block_size == 0

    def physical_block_for_token(self, token_index: int) -> int:
        logical_idx = token_index // self.block_size
        return self.physical_blocks[logical_idx]


class KVCacheManager:
    """
    Owns the pool of physical blocks and hands them out to requests on demand.

    num_blocks is total physical capacity expressed in blocks rather than
    tokens, mirroring how a real serving engine computes KV cache capacity
    from free VRAM and hands it out in fixed pages.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free_blocks: List[int] = list(range(num_blocks))
        self._block_tables: Dict[str, BlockTable] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def blocks_needed_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed_for_tokens(num_tokens) <= self.num_free_blocks

    def has_request(self, request_id: str) -> bool:
        return request_id in self._block_tables

    def allocate(self, request_id: str, num_tokens: int) -> BlockTable:
        """Reserve enough blocks to hold num_tokens for a brand new request."""
        if request_id in self._block_tables:
            raise ValueError(f"request {request_id} already has a block table")

        needed = self.blocks_needed_for_tokens(num_tokens)
        if needed > self.num_free_blocks:
            raise OutOfMemoryError(
                f"need {needed} blocks for {num_tokens} tokens, "
                f"only {self.num_free_blocks} free"
            )

        table = BlockTable(request_id=request_id, block_size=self.block_size)
        for _ in range(needed):
            table.physical_blocks.append(self._free_blocks.pop())
        table.num_tokens = num_tokens
        self._block_tables[request_id] = table
        return table

    def append_token(self, request_id: str) -> Optional[int]:
        """
        Grow a request's cache by one token (a single decode step).

        Returns the newly allocated physical block id if the append forced a
        fresh block, or None if the token fit into an already-allocated block.
        """
        table = self._block_tables.get(request_id)
        if table is None:
            raise KeyError(f"no block table for request {request_id}")

        needs_new_block = table.num_tokens == 0 or table.is_last_block_full()
        new_block_id = None
        if needs_new_block:
            if not self._free_blocks:
                raise OutOfMemoryError("no free blocks to extend sequence")
            new_block_id = self._free_blocks.pop()
            table.physical_blocks.append(new_block_id)

        table.num_tokens += 1
        return new_block_id

    def append_chunk(self, request_id: str, num_new_tokens: int) -> int:
        """
        Grow a request's cache by num_new_tokens in one shot (a prefill chunk).
        Returns how many new physical blocks were allocated.
        """
        if num_new_tokens <= 0:
            return 0
        table = self._block_tables.get(request_id)
        if table is None:
            raise KeyError(f"no block table for request {request_id}")

        target_tokens = table.num_tokens + num_new_tokens
        blocks_needed_total = self.blocks_needed_for_tokens(target_tokens)
        blocks_to_add = blocks_needed_total - len(table.physical_blocks)

        if blocks_to_add > self.num_free_blocks:
            raise OutOfMemoryError(
                f"need {blocks_to_add} more blocks, only {self.num_free_blocks} free"
            )

        for _ in range(blocks_to_add):
            table.physical_blocks.append(self._free_blocks.pop())
        table.num_tokens = target_tokens
        return blocks_to_add

    def can_append_tokens(self, request_id: str, num_new_tokens: int) -> bool:
        """Whether num_new_tokens can be appended to an existing request right now."""
        return self.max_appendable_tokens(request_id, num_new_tokens) >= num_new_tokens

    def max_appendable_tokens(self, request_id: str, desired: int) -> int:
        """
        How many of the desired new tokens can actually be appended given
        current free capacity -- lets the scheduler shrink a prefill chunk
        to fit instead of failing outright.
        """
        table = self._block_tables.get(request_id)
        if table is None:
            raise KeyError(f"no block table for request {request_id}")
        if desired <= 0:
            return 0

        capacity_in_allocated_blocks = len(table.physical_blocks) * self.block_size - table.num_tokens
        if desired <= capacity_in_allocated_blocks:
            return desired

        overflow = desired - capacity_in_allocated_blocks
        max_overflow = min(overflow, self.num_free_blocks * self.block_size)
        return capacity_in_allocated_blocks + max_overflow

    def free(self, request_id: str) -> None:
        table = self._block_tables.pop(request_id, None)
        if table is None:
            return
        self._free_blocks.extend(table.physical_blocks)

    def get_block_table(self, request_id: str) -> BlockTable:
        return self._block_tables[request_id]

    def usage(self) -> dict:
        used = self.num_blocks - self.num_free_blocks
        return {
            "num_blocks": self.num_blocks,
            "used_blocks": used,
            "free_blocks": self.num_free_blocks,
            "utilization_pct": round(100 * used / self.num_blocks, 2),
        }
