"""Nonpreemptive shared I/O transactions with foreground priority and write credit."""

from dataclasses import dataclass

import simpy

from TokenSim.config.validation import integer


@dataclass(frozen=True)
class QueueLimits:
    read_priority: bool = False
    write_bytes_per_second: float | None = None
    write_burst_bytes: int = 0
    max_backlog_bytes: int | None = None


class WriteCredit:
    def __init__(self, env, rate, burst):
        self.env = env
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.updated = env.now
        self.lock = simpy.Resource(env, capacity=1)

    def acquire(self, nbytes):
        if self.rate is None or not nbytes:
            return
        if nbytes > self.burst:
            raise ValueError("Write transaction exceeds its token burst")
        with self.lock.request() as request:
            yield request
            self.tokens = min(
                self.burst, self.tokens + (self.env.now - self.updated) * self.rate
            )
            self.updated = self.env.now
            if self.tokens < nbytes:
                delay = (nbytes - self.tokens) / self.rate
                yield self.env.timeout(delay)
                self.tokens = min(self.burst, self.tokens + delay * self.rate)
                self.updated = self.env.now
            self.tokens = max(0.0, self.tokens - nbytes)


class StorageQueue:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.resource = simpy.PriorityResource(env, capacity=1)
        self.credit = WriteCredit(
            env, policy.write_bytes_per_second, policy.write_burst_bytes
        )
        self.backlog_bytes = 0
        self.peak_backlog_bytes = 0
        self.sequence = 0
        self.events = []
        self.dropped_write_bytes = 0

    def reserve_write(self, nbytes):
        integer(nbytes, "queued write bytes")
        limit = self.policy.max_backlog_bytes
        if limit is not None and self.backlog_bytes + nbytes > limit:
            self.dropped_write_bytes += nbytes
            return False
        self.backlog_bytes += nbytes
        self.peak_backlog_bytes = max(self.peak_backlog_bytes, self.backlog_bytes)
        return True

    def execute(
        self, kind, nbytes, operation, *, prefetch=False, completed=None, context=None
    ):
        integer(nbytes, "I/O transaction bytes")
        write = kind == "write"
        queued = self.env.now
        try:
            if write:
                yield self.env.process(self.credit.acquire(nbytes))
            eligible = self.env.now
            priority = (
                (2 if write else 1 if prefetch else 0)
                if self.policy.read_priority
                else 0
            )
            serial = self.sequence
            self.sequence += 1
            with self.resource.request(priority=priority) as request:
                yield request
                started = self.env.now
                duration = operation()
                yield self.env.timeout(duration)
                if completed is not None:
                    completed()
                self.events.append(
                    dict(
                        sequence=serial,
                        kind=kind,
                        prefetch=prefetch,
                        bytes=nbytes,
                        queued_at=queued,
                        credit_ready_at=eligible,
                        started_at=started,
                        completed_at=self.env.now,
                        service_seconds=duration,
                        credit_wait_seconds=eligible - queued,
                        queue_wait_seconds=started - eligible,
                        **dict(context or {}),
                    )
                )
        finally:
            if write:
                self.backlog_bytes -= nbytes

    def describe(self):
        return dict(
            backlog_bytes=self.backlog_bytes,
            peak_backlog_bytes=self.peak_backlog_bytes,
            dropped_write_bytes=self.dropped_write_bytes,
            transactions=len(self.events),
            write_budget_scope="All submitted KV writeback bytes, an upper bound on NAND programs",
        )
