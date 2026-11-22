from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Generic, TypeVar

from TokenSim.errors import ConfigurationError
from TokenSim.utils import BoundedCache


WorkerT = TypeVar("WorkerT")


def replica_key(worker) -> tuple[str | None, int]:
    return getattr(worker, "execution_domain", None), getattr(worker, "dp_rank", 0)


class WorkerPool(Generic[WorkerT]):
    def __init__(self, workers: list[WorkerT]):
        if not workers:
            raise ConfigurationError("worker pool requires at least one worker")
        self.workers = workers
        self.start_idx = workers[0].id
        self.placement_decision_count = 0
        self.placements = Counter()

    def schedule(self) -> WorkerT:
        raise NotImplementedError

    def select_prefill_worker(self, request=None) -> WorkerT:
        return self._select()

    def select_decode_worker(
        self, request=None, *, preserve_dp_rank: bool = True
    ) -> WorkerT:
        _ = preserve_dp_rank
        return self._select()

    def select_transfer_target(
        self, requests=None, *, preserve_dp_rank: bool = True
    ) -> WorkerT:
        _ = preserve_dp_rank
        return self._select()

    def _select(self) -> WorkerT:
        worker = self.schedule()
        self.record_placement(worker)
        return worker

    def record_placement(self, worker):
        self.placement_decision_count += 1
        self.placements[replica_key(worker)] += 1

    def reset_stats(self):
        self.placement_decision_count = 0
        self.placements.clear()

    def placements_in_domain(self, domain):
        return {
            rank: self.placements[domain, rank]
            for worker in self.workers
            for worker_domain, rank in [replica_key(worker)]
            if worker_domain == domain
        }

    def get(self, id: int) -> WorkerT:
        return self.workers[id - self.start_idx]

    def __len__(self) -> int:
        return len(self.workers)


class DataParallelWorkerPool(WorkerPool[WorkerT]):
    def __init__(
        self,
        workers: list[WorkerT],
        pool_cls: type[WorkerPool[WorkerT]],
        *,
        sticky_sessions: bool = False,
    ):
        super().__init__(workers)
        grouped: dict[tuple[str | None, int], list[WorkerT]] = {}
        for worker in workers:
            grouped.setdefault(replica_key(worker), []).append(worker)
        self.group_pools = {key: pool_cls(values) for key, values in grouped.items()}
        self.replica_keys = sorted(
            self.group_pools, key=lambda key: (key[0] or "", key[1])
        )
        self.cur_dp_idx = 0
        self.sticky_sessions = sticky_sessions
        self.affinity_to_rank = BoundedCache(65536)

    @property
    def dp_placement_counts(self):
        return self.placements_in_domain(None)

    def schedule(self) -> WorkerT:
        return self._next_group_pool().schedule()

    def select_prefill_worker(self, request=None) -> WorkerT:
        return self._select_for_request(request, assign_new=True, preserve_dp_rank=True)

    def select_decode_worker(
        self, request=None, *, preserve_dp_rank: bool = True
    ) -> WorkerT:
        return self._select_for_request(
            request, assign_new=False, preserve_dp_rank=preserve_dp_rank
        )

    def select_transfer_target(
        self, requests=None, *, preserve_dp_rank: bool = True
    ) -> WorkerT:
        return self.select_decode_worker(
            requests[0] if requests else None, preserve_dp_rank=preserve_dp_rank
        )

    def _select_for_request(
        self, request, *, assign_new: bool, preserve_dp_rank: bool = True
    ) -> WorkerT:
        rank = getattr(request, "dp_rank", None) if preserve_dp_rank else None
        domain = getattr(request, "dp_domain", None)
        affinity_key = (
            getattr(request, "chat_id", None) if self.sticky_sessions else None
        )
        affinity_key = str(affinity_key) if affinity_key else None
        preferred = (domain, rank)
        if preferred not in self.group_pools and affinity_key is not None:
            preferred = self.affinity_to_rank.get(affinity_key)
        pool = self.group_pools.get(preferred)
        worker = (pool if pool is not None else self._next_group_pool()).schedule()
        self.record_placement(worker)
        selected = replica_key(worker)
        if request is not None and assign_new:
            request.dp_domain, request.dp_rank = selected
        if affinity_key is not None:
            self.affinity_to_rank[affinity_key] = selected
        return worker

    def _next_group_pool(self) -> WorkerPool[WorkerT]:
        key = self.replica_keys[self.cur_dp_idx]
        self.cur_dp_idx = (self.cur_dp_idx + 1) % len(self.replica_keys)
        return self.group_pools[key]


class RoundRobinWorkerPool(WorkerPool[WorkerT]):
    def __init__(self, workers: list[WorkerT]):
        super().__init__(workers)
        self.cur_idx = 0
        self.num_workers = len(workers)

    def schedule(self) -> WorkerT:
        i = self.cur_idx
        self.cur_idx = (self.cur_idx + 1) % self.num_workers
        return self.workers[i]


class LeastGpuMemoryWorkerPool(WorkerPool[WorkerT]):
    def workloads(self) -> list[float]:
        return [worker.workload() for worker in self.workers]

    def schedule(self) -> WorkerT:
        workloads = self.workloads()
        i = workloads.index(min(workloads))
        return self.workers[i]


@dataclass(frozen=True, order=True)
class WorkerLoad:
    active_requests: int
    gpu_memory_percent: float
    cpu_transfer_gb: float
    worker_id: int


class BalancedLoadWorkerPool(WorkerPool[WorkerT]):
    def worker_load(self, worker: WorkerT) -> WorkerLoad:
        scheduler = worker.scheduler
        active_requests = (
            len(scheduler.running)
            + len(scheduler.waiting)
            + len(getattr(scheduler, "loading", ()))
            + len(getattr(scheduler, "loaded_waiting", ()))
            + getattr(worker, "pending_admissions", 0)
        )
        return WorkerLoad(
            active_requests=active_requests,
            gpu_memory_percent=worker.workload(),
            cpu_transfer_gb=worker.cpu_workload(),
            worker_id=worker.id,
        )

    def schedule(self) -> WorkerT:
        return min(
            self.workers,
            key=lambda worker: self.worker_load(worker),
        )
