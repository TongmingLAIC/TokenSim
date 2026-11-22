from __future__ import annotations

from typing import Any, Tuple

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.block_table import BlockTable
from TokenSim.block.kv_cache_manager import KVCacheManager, PrefixReusePlan
from TokenSim.block.pool.allocator import BlockAllocator
from TokenSim.errors import SimulationStateError
from TokenSim.llm.llm_request import Request


class BlockManager:
    """Uniform paged KV blocks with prefix reuse."""

    grouped_cache = False

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int = 0,
        model: str = "unknown",
        watermark: float = 0.01,
    ):
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self._reserved_gpu_blocks: list[PhysicalTokenBlock] = []
        self.transfer_reservations: dict[int, int] = {}
        self.transfer_references: dict[Any, Any] = {}
        self.pd_handoff: Any = None
        self.block_table = BlockTable()
        self.kv_cache_manager = KVCacheManager(block_size=block_size, model=model)

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(
            Device.GPU,
            block_size,
            num_gpu_blocks,
            kv_cache_manager=self.kv_cache_manager,
        )

    def plan_reuse(self, req: Request) -> PrefixReusePlan:
        return self.kv_cache_manager.plan_reuse(req)

    def can_allocate(
        self, req: Request, *, plan: PrefixReusePlan | None = None
    ) -> bool:
        if plan is None:
            plan = self.plan_reuse(req)
        num_required_blocks = plan.miss_block_count
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_free_gpu_blocks += self.transfer_reservations.get(req.id, 0)
        num_free_gpu_blocks -= plan.zero_ref_hit_blocks
        return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, req: Request, *, plan: PrefixReusePlan | None = None):
        if self.block_table.get_num_blocks(req.id):
            raise SimulationStateError(
                f"request {req.id} already has blocks in this block manager"
            )
        transferred = self.transfer_reservations.pop(req.id, 0)
        if transferred:
            self.release_reserved_blocks(transferred)
        if plan is None:
            plan = self.plan_reuse(req)
        blocks: list[PhysicalTokenBlock] = []
        try:
            for block in plan.hit_blocks:
                self.gpu_allocator.acquire(block)
                blocks.append(block)
            for _ in range(plan.miss_block_count):
                blocks.append(self.gpu_allocator.allocate())
        except Exception:
            for block in reversed(blocks):
                self.gpu_allocator.free(block)
            raise
        self.kv_cache_manager.apply_plan(req, plan)
        self.block_table.add_blocks(req.id, blocks)
        self._sync_request_blocks(req, blocks)
        self._mark_input_blocks_pending(req, blocks, plan)
        if transferred and self.pd_handoff is not None:
            self.pd_handoff.admitted(req)

    def get_num_free_blocks(self):
        return self.gpu_allocator.get_num_free_blocks()

    def enable_retention(self, priority, clock):
        return self.gpu_allocator.pool.enable_retention(priority, clock)

    def get_num_allocated_blocks(self):
        return self.gpu_allocator.get_num_allocated_blocks()

    def get_gpu_status(self) -> Tuple[int, int, int]:
        return self.gpu_allocator.get_status()

    def get_cpu_status(self) -> Tuple[int, int, int]:
        return self.num_total_cpu_blocks, 0, self.num_total_cpu_blocks

    def get_num_blocks(self, req: Request) -> int:
        return self.block_table.get_num_blocks(req.id) or req.num_physical_token_blocks

    def free(self, req: Request):
        self.commit_finished_cache(req)
        self.release_request_blocks(req)

    def commit_finished_cache(self, req: Request) -> None:
        self.commit_input_cache(req)
        self.kv_cache_manager.register_output_blocks(
            req,
            self.block_table.get_blocks(req.id),
        )

    def release_request_blocks(self, req: Request) -> None:
        blocks = self.block_table.pop_blocks(req.id) or list(req._physical_token_blocks)
        for block in blocks:
            self.gpu_allocator.free(block)
        req._physical_token_blocks.clear()
        self.kv_cache_manager.forget_request(req.id)

    def can_append_slot(self, req: Request, reserved_blocks: int = 0) -> bool:
        required_blocks = int(
            self.block_table.get_num_blocks(req.id) < req.num_logical_token_blocks
        )
        return (
            self.gpu_allocator.get_num_free_blocks()
            >= required_blocks + reserved_blocks
        )

    def append_slot(self, req: Request):
        if self.block_table.get_num_blocks(req.id) < req.num_logical_token_blocks:
            block = self.gpu_allocator.allocate()
            self.block_table.add_block(req.id, block)
            req._append_physical_block(block)

    def rollback_last_block(self, req: Request) -> None:
        block = self.block_table.pop_last_block(req.id)
        if block is None:
            return
        self.gpu_allocator.free(block)
        self._sync_request_blocks(req, self.block_table.get_blocks(req.id))

    def get_request_block_counts(self, requests: list[Request]) -> dict[int, int]:
        return {req.id: self.get_num_blocks(req) for req in requests}

    def reserve_gpu_blocks(self, num_blocks: int):
        allocated: list[PhysicalTokenBlock] = []
        for _ in range(num_blocks):
            try:
                block = self.gpu_allocator.allocate()
            except Exception:
                for block in allocated:
                    self.gpu_allocator.free(block)
                raise
            allocated.append(block)
        self._reserved_gpu_blocks.extend(allocated)

    def release_reserved_blocks(self, num_blocks: int | None = None):
        if num_blocks is None:
            num_blocks = len(self._reserved_gpu_blocks)
        if num_blocks < 0:
            raise SimulationStateError(
                f"cannot release a negative block count: {num_blocks}"
            )
        if num_blocks > len(self._reserved_gpu_blocks):
            raise SimulationStateError(
                f"cannot release {num_blocks} reserved GPU blocks; "
                + f"only {len(self._reserved_gpu_blocks)} are reserved"
            )
        for _ in range(num_blocks):
            self.gpu_allocator.free(self._reserved_gpu_blocks.pop(0))

    @staticmethod
    def _sync_request_blocks(req: Request, blocks: list[PhysicalTokenBlock]) -> None:
        req._physical_token_blocks.clear()
        for block in blocks:
            req._append_physical_block(block)

    def _mark_input_blocks_pending(
        self,
        req: Request,
        blocks: list[PhysicalTokenBlock],
        plan: PrefixReusePlan,
    ) -> None:
        if not plan.input_keys:
            req.input_cache_keys = []
            return
        req.input_cache_keys = list(plan.input_keys)
        for index, key in enumerate(plan.input_keys):
            if index < plan.hit_block_count or index >= len(blocks):
                continue
            block = blocks[index]
            block.block_hash = key
            block.is_full = True
            block.cached = False

    def commit_input_cache(self, req: Request) -> None:
        if req.input_cache_committed:
            return
        blocks = self.block_table.get_blocks(req.id)
        if req.input_cache_keys:
            self.kv_cache_manager.register_blocks(
                blocks[: len(req.input_cache_keys)], req.input_cache_keys
            )
        req.input_cache_committed = True
