"""LRU ordering within evictable priority levels, with lazy TTL expiration."""

from collections import OrderedDict
import heapq
import math


class PriorityFreeMap(OrderedDict):
    def __init__(self, priority, clock, order=None):
        super().__init__()
        self.priority, self.clock = priority, clock
        self.order = order
        self._sequence = 0
        self._version = 0
        self._stamps = {}
        self._versions = {}
        self._heap = []
        self._expiry = []

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._sequence += 1
        self._stamps[key] = (
            self.order(key, value) if self.order else self._sequence,
            self._sequence,
        )
        self.refresh(key)

    def __delitem__(self, key):
        super().__delitem__(key)
        self._stamps.pop(key, None)
        self._versions.pop(key, None)

    def refresh(self, key):
        if key not in self:
            return
        priority, expiry = self.priority(key)
        if not 0 <= priority <= 100 or expiry <= self.clock():
            raise ValueError("Priority provider returned invalid priority or TTL")
        self._version += 1
        version = self._versions[key] = self._version
        heapq.heappush(self._heap, (priority, self._stamps[key], version, key))
        if math.isfinite(expiry):
            heapq.heappush(self._expiry, (expiry, version, key))
        if len(self._heap) > 4 * len(self) + 1024:
            self._heap = [
                row for row in self._heap if self._versions.get(row[3]) == row[2]
            ]
            heapq.heapify(self._heap)

    def _expire(self):
        while self._expiry and self._expiry[0][0] <= self.clock():
            _, version, key = heapq.heappop(self._expiry)
            if self._versions.get(key) == version:
                self.refresh(key)

    def popitem(self, last=True):
        if not self:
            raise KeyError("empty priority queue")
        if last:
            key = next(reversed(self))
        else:
            self._expire()
            while self._heap:
                _, _, version, key = heapq.heappop(self._heap)
                if self._versions.get(key) == version:
                    break
            else:
                raise RuntimeError("Priority queue lost an evictable entry")
        value = self[key]
        del self[key]
        return key, value

    def pop(self, key, *default):
        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        value = self[key]
        del self[key]
        return value

    def clear(self):
        super().clear()
        self._stamps.clear()
        self._versions.clear()
        self._heap.clear()
        self._expiry.clear()
