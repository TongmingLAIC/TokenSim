"""Group-aware external storage and queued DMA primitives for native KV."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from heapq import heappop, heappush, heapify

import simpy

from TokenSim.errors import SimulationStateError
from TokenSim.config.validation import integer, number


@dataclass
class StoredPage:
    key: tuple
    bytes: int
    references: int = 0
    ready: bool = False
    last_touch: int = 0
    eviction_token: int = 0


class NativePageStore:
    """Bounded LRU whose writes/reads pin pages until DMA completion.

    SimpleCPU supplies its packed backing stride as allocation_unit. Mooncake
    supplies actual per-group object bytes and uses one shared node budget.
    """

    def __init__(
        self, capacity_bytes, *, allocation_unit=None, scope="shared", before_evict=None
    ):
        self.capacity_bytes = integer(capacity_bytes, "capacity_bytes", minimum=1)
        if allocation_unit is not None:
            integer(allocation_unit, "allocation_unit", minimum=1)
        self.allocation_unit = allocation_unit
        self.scope = scope
        self.before_evict = before_evict
        self._unpinned_count = 0
        self.mutation_version = 0
        self._touch_clock = self._eviction_sequence = 0
        self._evictable = None
        self.pages = {}
        self.lru = OrderedDict()
        self.used_bytes = self.evictions = self.peak_used_bytes = 0

    def allocation_bytes(self, payload_bytes):
        return self.allocation_unit or payload_bytes

    def lookup(self, key):
        page = self.pages.get(key)
        if page is None or not page.ready:
            return None
        self._touch(page)
        return page

    def _offer_eviction(self, page):
        if self._evictable is None:
            return
        self._eviction_sequence += 1
        page.eviction_token = self._eviction_sequence
        heappush(self._evictable, (page.last_touch, page.eviction_token, page.key))
        if len(self._evictable) > max(1024, 4 * len(self.pages)):
            self._evictable = [
                (p.last_touch, p.eviction_token, p.key)
                for p in self.pages.values()
                if p.references == 0
            ]
            heapify(self._evictable)

    def _touch(self, page):
        self._touch_clock += 1
        page.last_touch = self._touch_clock
        self.lru.move_to_end(page.key)
        if page.references == 0:
            self._offer_eviction(page)

    def _victim(self, vetoed):
        if self._evictable is None:
            self._evictable = []
            for page in self.pages.values():
                if page.references == 0:
                    self._offer_eviction(page)
        while self._evictable:
            touched, token, key = heappop(self._evictable)
            page = self.pages.get(key)
            if (
                page is not None
                and not page.references
                and page.last_touch == touched
                and page.eviction_token == token
                and key not in vetoed
            ):
                return page
        return None

    def reserve_write(self, key, payload_bytes):
        integer(payload_bytes, "payload_bytes", minimum=1)
        if key in self.pages:
            return None
        size = self.allocation_bytes(payload_bytes)
        if size > self.capacity_bytes:
            return None
        vetoed, deferred = set(), []
        try:
            while self.used_bytes + size > self.capacity_bytes:
                if self._unpinned_count == 0:
                    return None
                old = self._victim(vetoed)
                if old is None:
                    return None
                if self.before_evict is not None and not self.before_evict(old):
                    vetoed.add(old.key)
                    deferred.append(old)
                    continue
                if not self.discard(old):
                    raise SimulationStateError("an eviction approval pinned its victim")
            page = StoredPage(key, size, references=1)
            self.pages[key] = page
            self.mutation_version += 1
            self.lru[key] = None
            self._touch(page)
            self.used_bytes += size
            self.peak_used_bytes = max(self.peak_used_bytes, self.used_bytes)
            return page
        finally:
            for page in deferred:
                if self.pages.get(page.key) is page and page.references == 0:
                    self._offer_eviction(page)

    def discard(self, page):
        """Reclaim an unreferenced cache copy without changing another tier."""
        if self.pages.get(page.key) is not page or page.references:
            return False
        self.pages.pop(page.key)
        self.lru.pop(page.key)
        self._unpinned_count -= 1
        self.mutation_version += 1
        self.used_bytes -= page.bytes
        self.evictions += 1
        return True

    def finish_write(self, page):
        if self.pages.get(page.key) is not page:
            raise SimulationStateError(
                "a pinned external page was evicted during write"
            )
        page.ready = True
        self.release(page)

    def acquire(self, page, *, touch=True):
        if self.pages.get(page.key) is not page or not page.ready:
            raise SimulationStateError("external read needs a ready resident page")
        if page.references == 0:
            self._unpinned_count -= 1
        page.references += 1
        self.mutation_version += 1
        if touch:
            self._touch(page)

    def release(self, page):
        if page.references <= 0:
            raise SimulationStateError("invalid external page release")
        page.references -= 1
        self.mutation_version += 1
        if page.references == 0:
            self._unpinned_count += 1
            self._offer_eviction(page)

    def validate(self):
        if self._unpinned_count != sum(p.references == 0 for p in self.pages.values()):
            raise SimulationStateError(
                "external storage unpinned-page conservation failed"
            )
        if (
            self.used_bytes != sum(p.bytes for p in self.pages.values())
            or self.used_bytes > self.capacity_bytes
        ):
            raise SimulationStateError("external storage byte conservation failed")
        if set(self.pages) != set(self.lru) or any(
            p.references < 0 for p in self.pages.values()
        ):
            raise SimulationStateError(
                "external storage reference/LRU conservation failed"
            )
        return True

    def describe(self):
        return {
            "scope": self.scope,
            "capacity_bytes": self.capacity_bytes,
            "allocation_unit_bytes": self.allocation_unit,
            "used_bytes": self.used_bytes,
            "peak_used_bytes": self.peak_used_bytes,
            "resident_pages": len(self.pages),
            "pinned_pages": sum(p.references > 0 for p in self.pages.values()),
            "evictions": self.evictions,
        }


class QueuedKVLink:
    """One shared directional link; independent callers contend in one queue."""

    def __init__(self, env, bandwidth_bytes_per_s, *, utilization=1.0, latency_us=0):
        number(bandwidth_bytes_per_s, "bandwidth_bytes_per_s", positive=True)
        number(utilization, "utilization", positive=True, maximum=1)
        number(latency_us, "latency_us")
        self.env = env
        self.bandwidth_bytes_per_s = bandwidth_bytes_per_s
        self.utilization = utilization
        self.latency_seconds = latency_us / 1e6
        self.queue = simpy.Resource(env, capacity=1)
        self.bytes = self.jobs = 0
        self.queue_wait_seconds = self.service_seconds = 0.0

    def transfer(self, bytes_):
        integer(bytes_, "transfer bytes")
        issued = self.env.now
        with self.queue.request() as slot:
            yield slot
            self.queue_wait_seconds += self.env.now - issued
            duration = self.latency_seconds + bytes_ / (
                self.bandwidth_bytes_per_s * self.utilization
            )
            self.jobs += 1
            self.bytes += bytes_
            self.service_seconds += duration
            yield self.env.timeout(duration)

    def describe(self):
        return {
            "theoretical_bandwidth_bytes_per_s": self.bandwidth_bytes_per_s,
            "utilization_assumption": self.utilization,
            "bytes": self.bytes,
            "jobs": self.jobs,
            "queue_wait_seconds": self.queue_wait_seconds,
            "service_seconds": self.service_seconds,
        }


def validate_external_stores(connectors):
    """Validate each distinct owner of storage invariants once after drain.

    A view can expose validation_owner when its validate method delegates
    entirely to that owner and adds no client-local invariant checks.
    """
    validated = set()
    for connector in connectors:
        store = connector.store
        owner = getattr(store, "validation_owner", store)
        if id(owner) not in validated:
            store.validate()
            validated.add(id(owner))
    return len(validated)
