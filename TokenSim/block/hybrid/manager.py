"""One physical pool coordinating full and sliding-window cache groups."""

from __future__ import annotations

from collections import OrderedDict

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.config.validation import integer
from TokenSim.errors import OutOfBlocksError
from .allocator import SlotAllocatorMixin
from .external_cache import ExternalCacheMixin
from .kv_layout import HybridKVLayout
from .handoff import GroupedHandoffMixin
from .prefix_cache import PrefixCacheMixin


class _GroupSpec:
    """Per-group constants read on every scheduling step."""

    __slots__ = (
        "group",
        "group_id",
        "block_tokens",
        "window_tokens",
        "eagle",
        "need",
        "max_request_slots",
    )

    def __init__(self, group, layout):
        self.group = group
        self.group_id = group.group_id
        self.block_tokens = group.block_tokens
        self.window_tokens = group.window_tokens
        self.eagle = int(group.eagle)
        self.need = group.hit_tail_slots
        self.max_request_slots = group.max_request_slots(
            layout.max_model_len, layout.max_in_flight_tokens
        )


class _HybridBlockTable:
    """Request-to-slot view derived from the group tables in layout order."""

    def __init__(self, manager):
        self._manager = manager

    def get_blocks(self, request_id):
        tables = self._manager.tables.get(request_id)
        if not tables:
            return []
        return [block for table in tables.values() for block in table.values()]

    def get_num_blocks(self, request_id):
        tables = self._manager.tables.get(request_id)
        return sum(len(table) for table in tables.values()) if tables else 0


class HybridKVBlockManager(
    PrefixCacheMixin, SlotAllocatorMixin, ExternalCacheMixin, GroupedHandoffMixin
):
    """Group-aware slot allocation with atomic prefix recovery and retention."""

    grouped_cache = True

    def __init__(
        self, layout: HybridKVLayout, num_slots: int, model: str, *, watermark=0.0
    ):
        num_slots = integer(num_slots, "hybrid physical slot count")
        if num_slots < 2:
            raise OutOfBlocksError(
                "native hybrid pool needs a null slot and usable slots"
            )
        self.layout, self.model = layout, model
        self.block_size = layout.logical_block_tokens
        self.num_total_gpu_blocks = num_slots
        self.num_total_cpu_blocks = 0
        self.watermark_blocks = int(num_slots * watermark)
        self.blocks = [
            PhysicalTokenBlock(Device.GPU, i, self.block_size) for i in range(num_slots)
        ]
        self.null_block = self.blocks[0]
        self.null_block.ref_count = 1
        self._free_uncached = self.blocks[:0:-1]
        self._free_cached = OrderedDict()
        self.cache = {}
        self.duplicates = {}
        self.cached_progress = {}
        self.tables = {}
        self.requests = {}
        self.keys = {}
        self.fine_keys = {}
        self.transfer_references = {}
        self.transfer_reservations = {}
        self.pd_handoff = None
        self.publish_callbacks = []
        self.block_table = _HybridBlockTable(self)
        self.kv_cache_manager = self
        self.gpu_allocator = self
        self.allocation_count = self.release_count = self.eviction_count = 0
        self.peak_used_slots = 1
        self.groups = tuple(_GroupSpec(group, layout) for group in layout.groups)
        self.full_spec = next(
            spec for spec in self.groups if spec.group is layout.full_group
        )
        self._full_growth = tuple(
            (spec.group_id, spec.block_tokens)
            for spec in self.groups
            if not spec.window_tokens
        )
        self._window_growth = tuple(
            (position, spec.group_id, spec.block_tokens, spec.window_tokens - 1)
            for position, spec in enumerate(self.groups)
            if spec.window_tokens
        )
        interval = layout.retention_interval
        self._tails_fit_segments = not interval or all(
            spec.need * spec.block_tokens <= interval for spec in self.groups
        )
        self._cache_version = 0
        self._plans = {}
        self._committed = {}
        self._spans = {}

    def get_cpu_status(self) -> tuple[int, int, int]:
        return 0, 0, 0
