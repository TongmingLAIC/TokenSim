"""AF attention replica KV cache coordinating per-rank pools."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Mapping

from TokenSim.af.runtime.plan import AFRuntimePlan
from TokenSim.af.runtime.records import AFReplicaRuntime, RankKey
from TokenSim.block.block_manager import BlockManager
from TokenSim.block.kv_cache_manager import PrefixReusePlan
from TokenSim.errors import ConfigurationError, OutOfBlocksError, SimulationStateError
from TokenSim.llm.llm_request import Request


class ReplicaKVBlockManager:
    """One scheduler-visible KV cache backed by one local pool per A TP rank."""

    grouped_cache = False

    def __init__(
        self,
        *,
        block_size: int,
        model: str,
        replica: AFReplicaRuntime,
        gpu_blocks_by_rank: Mapping[RankKey, int] | None = None,
        cpu_blocks_by_rank: Mapping[RankKey, int] | None = None,
    ) -> None:
        if not replica.is_attention or replica.leader_rank_key is None:
            raise ConfigurationError(
                f"replica KV manager requires an A replica, got {replica.domain!r}"
            )
        self.block_size = block_size
        self.model = model
        self.replica = replica
        self.replica_key = replica.replica_key
        self.domain = replica.domain
        self.instance = replica.instance
        self.dp_rank = replica.dp_rank
        self.rank_keys = tuple(replica.rank_keys)
        self.leader_rank_key = replica.leader_rank_key
        self.owner_instances = {replica.domain: replica.instance}
        gpu_capacities = dict(gpu_blocks_by_rank or replica.local_gpu_blocks)
        cpu_capacities = dict(cpu_blocks_by_rank or replica.local_cpu_blocks)
        if set(gpu_capacities) != set(self.rank_keys):
            raise ConfigurationError(
                f"AF replica {self.replica_key} GPU capacities do not match ranks: "
                f"expected={self.rank_keys}, actual={tuple(gpu_capacities)}"
            )
        if set(cpu_capacities) != set(self.rank_keys):
            raise ConfigurationError(
                f"AF replica {self.replica_key} CPU capacities do not match ranks: "
                f"expected={self.rank_keys}, actual={tuple(cpu_capacities)}"
            )
        self.transfer_reservations: dict[int, int] = {}
        self.pd_handoff: Any = None
        self.rank_managers = {
            rank_key: BlockManager(
                block_size=block_size,
                num_gpu_blocks=gpu_capacities[rank_key],
                num_cpu_blocks=cpu_capacities[rank_key],
                model=model,
                watermark=0,
            )
            for rank_key in self.rank_keys
        }
        for manager in self.rank_managers.values():
            manager.transfer_reservations = self.transfer_reservations
        self.num_total_gpu_blocks = min(gpu_capacities.values())
        self.num_total_cpu_blocks = min(cpu_capacities.values())
        self.watermark_blocks = 0
        leader = self.rank_managers[self.leader_rank_key]
        self.block_table = leader.block_table
        self.kv_cache_manager = leader.kv_cache_manager
        self.gpu_allocator = leader.gpu_allocator
        self._request_views: dict[int, dict[RankKey, Request]] = {}

    @classmethod
    def from_plan(
        cls,
        plan: AFRuntimePlan,
        replica: AFReplicaRuntime,
    ) -> "ReplicaKVBlockManager":
        return cls(block_size=plan.block_size, model=plan.model, replica=replica)

    def _validate_request_owner(self, req: Request, *, set_if_missing: bool) -> None:
        current = req.af_a_owners.get(self.domain)
        if current is not None and current != self.instance:
            raise SimulationStateError(
                f"request {req.id} has AF A owner conflict in {self.domain}: "
                f"expected {self.instance}, got {current}"
            )
        if set_if_missing:
            req.af_a_owners[self.domain] = self.instance

    def _make_view(self, rank_key: RankKey, req: Request) -> Request:
        if rank_key == self.leader_rank_key:
            return req
        view = copy.copy(req)
        view._physical_token_blocks = []
        view.input_cache_keys = list(req.input_cache_keys)
        return view

    def _require_views(self, req: Request) -> dict[RankKey, Request]:
        self._validate_request_owner(req, set_if_missing=False)
        try:
            views = self._request_views[req.id]
        except KeyError as exc:
            raise SimulationStateError(
                f"request {req.id} has no allocated shards in AF replica {self.replica_key}"
            ) from exc
        for rank_key, view in views.items():
            if rank_key != self.leader_rank_key:
                view._logical_tokens = req._logical_tokens
                view.generation_idx = req.generation_idx
                view.num_computed_tokens = req.num_computed_tokens
        return views

    def _rank_reuse_plans(self, req: Request) -> dict[RankKey, PrefixReusePlan]:
        plans = {
            rank_key: manager.plan_reuse(req)
            for rank_key, manager in self.rank_managers.items()
        }
        shared = 0
        for shards in zip(*(plan.hit_blocks for plan in plans.values())):
            if len({block.ref_count for block in shards}) != 1:
                break
            shared += 1
        hit_tokens = shared * self.block_size
        return {
            rank_key: replace(
                plan,
                hit_blocks=plan.hit_blocks[:shared],
                hit_tokens=hit_tokens,
                effective_prefill_tokens=req.prefill_len - hit_tokens,
            )
            for rank_key, plan in plans.items()
        }

    def plan_reuse(self, req: Request) -> PrefixReusePlan:
        return self._rank_reuse_plans(req)[self.leader_rank_key]

    def can_allocate(self, req: Request) -> bool:
        self._validate_request_owner(req, set_if_missing=False)
        plans = self._rank_reuse_plans(req)
        return all(
            manager.can_allocate(req, plan=plans[rank_key])
            for rank_key, manager in self.rank_managers.items()
        )

    def allocate(self, req: Request) -> None:
        self._validate_request_owner(req, set_if_missing=False)
        if req.id in self._request_views:
            raise SimulationStateError(
                f"request {req.id} already has KV shards in AF replica {self.replica_key}"
            )
        transferred = self.transfer_reservations.pop(req.id, 0)
        if transferred:
            self.release_reserved_blocks(transferred)
        views = {
            rank_key: self._make_view(rank_key, req) for rank_key in self.rank_keys
        }
        plans = self._rank_reuse_plans(req)
        if not all(
            self.rank_managers[rank_key].can_allocate(
                views[rank_key], plan=plans[rank_key]
            )
            for rank_key in self.rank_keys
        ):
            raise OutOfBlocksError(
                f"AF replica {self.replica_key} cannot allocate request {req.id} "
                "on every TP rank"
            )
        state = req.admission_state()
        completed: list[RankKey] = []
        try:
            for rank_key in self.rank_keys:
                self.rank_managers[rank_key].allocate(
                    views[rank_key], plan=plans[rank_key]
                )
                completed.append(rank_key)
        except Exception:
            for rank_key in reversed(completed):
                self.rank_managers[rank_key].release_request_blocks(views[rank_key])
            req.restore_admission_state(state)
            req._physical_token_blocks.clear()
            raise
        self._request_views[req.id] = views
        self._validate_request_owner(req, set_if_missing=True)
        if transferred and self.pd_handoff is not None:
            self.pd_handoff.admitted(req)

    def can_append_slot(self, req: Request, reserved_blocks: int = 0) -> bool:
        views = self._require_views(req)
        return all(
            self.rank_managers[rank_key].can_append_slot(
                views[rank_key], reserved_blocks
            )
            for rank_key in self.rank_keys
        )

    def append_slot(self, req: Request) -> None:
        views = self._require_views(req)
        if not self.can_append_slot(req):
            raise OutOfBlocksError(
                f"AF replica {self.replica_key} cannot append request {req.id} "
                "on every TP rank"
            )
        old_counts = {
            rank_key: manager.block_table.get_num_blocks(req.id)
            for rank_key, manager in self.rank_managers.items()
        }
        completed: list[RankKey] = []
        try:
            for rank_key in self.rank_keys:
                self.rank_managers[rank_key].append_slot(views[rank_key])
                completed.append(rank_key)
        except Exception:
            for rank_key in reversed(completed):
                manager = self.rank_managers[rank_key]
                if manager.block_table.get_num_blocks(req.id) > old_counts[rank_key]:
                    manager.rollback_last_block(views[rank_key])
            raise

    def commit_input_cache(self, req: Request) -> None:
        views = self._require_views(req)
        for rank_key in self.rank_keys:
            self.rank_managers[rank_key].commit_input_cache(views[rank_key])

    def commit_finished_cache(self, req: Request) -> None:
        views = self._require_views(req)
        for rank_key, view in views.items():
            self.rank_managers[rank_key].commit_finished_cache(view)

    def free(self, req: Request) -> None:
        self.commit_finished_cache(req)
        self.release_request_blocks(req)

    def release_request_blocks(self, req: Request) -> None:
        views = self._require_views(req)
        for rank_key in reversed(self.rank_keys):
            self.rank_managers[rank_key].release_request_blocks(views[rank_key])
        self._request_views.pop(req.id, None)

    def reserve_gpu_blocks(self, num_blocks: int) -> None:
        completed: list[RankKey] = []
        try:
            for rank_key in self.rank_keys:
                self.rank_managers[rank_key].reserve_gpu_blocks(num_blocks)
                completed.append(rank_key)
        except Exception:
            for rank_key in reversed(completed):
                self.rank_managers[rank_key].release_reserved_blocks(num_blocks)
            raise

    def release_reserved_blocks(self, num_blocks: int | None = None) -> None:
        for rank_key in reversed(self.rank_keys):
            self.rank_managers[rank_key].release_reserved_blocks(num_blocks)

    def get_num_blocks(self, req: Request) -> int:
        return self.block_table.get_num_blocks(req.id) or req.num_physical_token_blocks

    def get_request_block_counts(self, requests: list[Request]) -> dict[int, int]:
        return {req.id: self.get_num_blocks(req) for req in requests}

    def get_gpu_status(self) -> tuple[int, int, int]:
        free = min(
            manager.gpu_allocator.get_num_free_blocks()
            for manager in self.rank_managers.values()
        )
        total = self.num_total_gpu_blocks
        return free, total - free, total

    def get_num_free_blocks(self) -> int:
        return min(
            manager.get_num_free_blocks() for manager in self.rank_managers.values()
        )

    def get_num_allocated_blocks(self) -> int:
        return max(
            manager.get_num_allocated_blocks()
            for manager in self.rank_managers.values()
        )

    def get_cpu_status(self) -> tuple[int, int, int]:
        return self.num_total_cpu_blocks, 0, self.num_total_cpu_blocks

    def rank_status(self, rank_key: RankKey) -> dict[str, int]:
        try:
            manager = self.rank_managers[rank_key]
        except KeyError as exc:
            raise ConfigurationError(
                f"AF replica {self.replica_key} has no rank {rank_key}"
            ) from exc
        _, gpu_used, gpu_total = manager.get_gpu_status()
        _, cpu_used, cpu_total = manager.get_cpu_status()
        return {
            "gpu_capacity": gpu_total,
            "gpu_used": gpu_used,
            "cpu_capacity": cpu_total,
            "cpu_used": cpu_used,
        }
