"""Allocate physical KV blocks and own their references."""

from typing import Tuple

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.pool.blocks import BlockPool
from TokenSim.block.kv_cache_manager import KVCacheManager
from TokenSim.errors import OutOfBlocksError, SimulationStateError


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
        kv_cache_manager: KVCacheManager | None = None,
    ) -> None:
        self.device = device
        self.block_size: int = block_size
        self.num_blocks: int = int(num_blocks)
        self.kv_cache_manager = kv_cache_manager
        self.pool = BlockPool(device, block_size, num_blocks)
        self.free_blocks = self.pool.free_blocks

    def allocate(self) -> PhysicalTokenBlock:
        if not self.get_num_free_blocks():
            raise OutOfBlocksError(
                f"out of {self.device.name} memory; no free blocks are available"
            )
        block = self.pool.pop_free_block()
        if block.cached:
            if self.kv_cache_manager is None:
                block.clear_cache_metadata()
            else:
                self.kv_cache_manager.evict(block)
        if block.ref_count != 0:
            raise SimulationStateError(
                f"{self.device.name} block {block.block_number} is free but "
                + f"has ref_count {block.ref_count}"
            )
        block.ref_count += 1
        return block

    def acquire(self, block: PhysicalTokenBlock) -> None:
        self.pool.validate_owned(block)
        if block.ref_count == 0:
            self.pool.remove_from_free_queue(block)
        block.ref_count += 1
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.touch(block)

    def free(self, block: PhysicalTokenBlock | None = None) -> None:
        if block is None:
            raise SimulationStateError(
                "BlockAllocator.free() requires a physical block"
            )
        self.pool.validate_owned(block)
        if block.ref_count <= 0:
            raise SimulationStateError(
                f"block {block.block_number} ref_count is already {block.ref_count}"
            )
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.release_block(block, self.free_blocks)
        else:
            block.ref_count -= 1
            if block.ref_count == 0:
                block.clear_cache_metadata()
                self.pool.append_free_block(block)

    def get_num_free_blocks(self) -> int:
        return self.pool.get_num_free_blocks()

    def get_num_allocated_blocks(self) -> int:
        return self.num_blocks - self.get_num_free_blocks()

    def get_status(self) -> Tuple[int, int, int]:
        return (
            self.get_num_free_blocks(),
            self.get_num_allocated_blocks(),
            self.num_blocks,
        )
