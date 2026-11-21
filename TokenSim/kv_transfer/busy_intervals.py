"""Chronological forward-interval union and bounded I/O-window overlap queries."""

from bisect import bisect_left, bisect_right

from TokenSim.errors import SimulationStateError


def append_union(intervals, start, end):
    if end < start:
        raise SimulationStateError("Forward busy interval has negative duration")
    if intervals:
        left, right = intervals[-1]
        if start < left:
            raise SimulationStateError(
                "Forward busy intervals are out of chronological order"
            )
        if start <= right:
            intervals[-1] = (left, max(right, end))
            return
    intervals.append((start, end))


class BusyIndex:
    def __init__(self, intervals):
        merged = []
        for start, end in intervals:
            append_union(merged, start, end)
        self.starts = [start for start, _ in merged]
        self.ends = [end for _, end in merged]

    def uncovered(self, start, end):
        if end <= start:
            return 0.0
        begin = bisect_right(self.ends, start)
        finish = bisect_left(self.starts, end)
        covered = 0.0
        # Clip only the intersecting union spans, retaining the original
        # local summation order and avoiding large-prefix cancellation.
        for index in range(begin, finish):
            covered += min(end, self.ends[index]) - max(start, self.starts[index])
        return max(0.0, end - start - covered)
