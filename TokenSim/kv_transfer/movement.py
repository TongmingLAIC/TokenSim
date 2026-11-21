"""Online tool-wait demotion and prefix preparation on native queued transfers."""

from dataclasses import dataclass
import simpy

from TokenSim.errors import OutOfBlocksError
from TokenSim.llm.llm_request import Request


@dataclass(frozen=True)
class PrefixSnapshot:
    owner: str
    manager: object
    connector: object
    prompt: int
    block_size: int
    hashes: tuple
    fine_hashes: tuple | None
    fine_block_size: int
    cache_salt: str | None
    reuse_group: str | None
    blocks: tuple

    @classmethod
    def capture(cls, req, manager, connector):
        fine = getattr(req, "trace_hash_ids", None)
        return cls(
            req.chat_id or f"request:{req.id}",
            manager,
            connector,
            req.prefill_len,
            req.block_size,
            tuple(req.hash_ids or ()),
            tuple(fine) if fine is not None else None,
            getattr(req, "trace_block_size", req.block_size),
            req.cache_salt,
            req.reuse_group,
            tuple(
                (block, block.block_hash)
                for block in manager.block_table.get_blocks(req.id)
                if block.block_hash is not None
            ),
        )

    def compatible(self, req):
        hashes = req.hash_ids or ()
        return (
            req.cache_salt == self.cache_salt
            and req.reuse_group == self.reuse_group
            and len(hashes) >= len(self.hashes)
            and tuple(hashes[: len(self.hashes)]) == self.hashes
        )


class MovementRuntime:
    def __init__(self, controller):
        self.controller, self.env = controller, controller.env
        self.snapshots = {}
        self.snapshot_revision = 0
        self.parent_content_cache = {}
        self.waiting = {}
        self.estimates = {}
        self.admitted_estimates = {}
        self.jobs = {}
        self._next_id = -(10**12)
        self.closed = False
        self.active = set()
        self._job_id = 0
        self.changed = self.env.event()
        self.stats = dict(
            move_pages=0,
            move_busy_pages=0,
            prefetch_started=0,
            prefetch_completed=0,
            prefetch_cancelled=0,
            prefetch_budget_rejected=0,
            prefetch_capacity_rejected=0,
            prefetch_bytes=0,
            foreground_wait_seconds=0.0,
        )
        self.blocked_since = {}

    def capture(self, scheduler, req):
        self.snapshot_revision += 1
        self.snapshots[self.controller.owner(req)] = PrefixSnapshot.capture(
            req, scheduler.block_manager, scheduler.connector
        )

    def tool_wait(self, req):
        owner = self.controller.owner(req)
        snapshot = self.snapshots.get(owner)
        if owner in self.jobs and self.jobs[owner]["phase"] == "finished":
            self.jobs.pop(owner)
        if self.closed or snapshot is None or owner in self.jobs:
            return
        self.waiting[owner] = self.env.now
        estimate = self.estimates.get(owner)
        mode = getattr(self.controller, "mode", "movement")
        if mode == "forced_movement":
            estimate = 30.0
        elif mode == "enhanced_coop" and getattr(
            self.controller, "queue_window_enabled", True
        ):
            queue = self.controller.queue_estimate(snapshot.connector.scheduler)
            estimate = max(
                self.admitted_estimates.get(owner, 0), (estimate or 0) + queue
            )
        self._job_id += 1
        state = dict(
            snapshot=snapshot,
            phase="moving",
            request=None,
            done=self.env.event(),
            job_id=self._job_id,
        )
        self.active.add(self._job_id)
        self.jobs[owner] = state
        state["process"] = self.env.process(self._wait(state, estimate))

    def _wait(self, state, estimate):
        snapshot = state["snapshot"]
        try:
            yield self.env.timeout(0)
            if estimate is None or estimate < 0.5:
                state["phase"] = "idle"
                return
            mode = getattr(self.controller, "mode", "movement")
            if mode in ("all", "enhanced_coop"):
                manager = snapshot.manager
                # Cached-but-evictable pages occupy physical HBM too.
                occupied = manager.num_total_gpu_blocks - len(manager._free_uncached)
                if occupied / manager.num_total_gpu_blocks < 0.8:
                    self.stats["occupancy_skipped"] = (
                        self.stats.get("occupancy_skipped", 0) + 1
                    )
                    return
            self.move_ready(snapshot)
            state["phase"] = "sleeping"
            lead = self.controller.engine.config.cache_prefetch_lead_seconds
            yield self.env.timeout(max(0, estimate - lead))
            if self.closed:
                return
            state["phase"] = "prefetch"
            yield self.env.process(self.copy(snapshot, state))
        except simpy.Interrupt:
            pass
        finally:
            state["phase"] = "finished"
            if not state["done"].triggered:
                state["done"].succeed()
            self.active.discard(state["job_id"])
            if not self.changed.triggered:
                self.changed.succeed()
            self.changed = self.env.event()
            snapshot.connector.notify()

    def move_ready(self, snapshot):
        manager, connector = snapshot.manager, snapshot.connector
        moved = busy = 0
        for block, tag in snapshot.blocks:
            if block.block_hash != tag:
                continue
            if connector.store.lookup(tag) is None or block.ref_count:
                busy += 1
                continue
            slot = block.block_number
            if slot not in manager._free_cached:
                continue
            del manager._free_cached[slot]
            manager._evict(block)
            manager._free_uncached.append(block)
            moved += 1
        self.stats["move_pages"] += moved
        self.stats["move_busy_pages"] += busy
        self.controller.record(
            "move_complete",
            owner=snapshot.owner,
            pages=moved,
            busy_or_unwritten_pages=busy,
        )
        return moved

    def copy(self, snapshot, state):
        manager, connector = snapshot.manager, snapshot.connector
        req = Request(
            self._next_id,
            snapshot.prompt,
            1,
            snapshot.block_size,
            chat_id=snapshot.owner,
            hash_ids=list(snapshot.hashes),
            cache_salt=snapshot.cache_salt,
            reuse_group=snapshot.reuse_group,
        )
        self._next_id -= 1
        if snapshot.fine_hashes is not None:
            req.trace_hash_ids = snapshot.fine_hashes
            req.trace_block_size = snapshot.fine_block_size
        plan = manager.plan_reuse(req)
        boundary, hits = connector.probe(req, plan)
        if not hits:
            manager.release_request_blocks(req)
            req.release_logical_blocks()
            return
        pages = [page for _, pairs in hits for _, page in pairs]
        size = connector._transfer_bytes(pages)
        if size > self.controller.engine.config.cache_prefetch_budget_bytes:
            self.stats["prefetch_budget_rejected"] += 1
            manager.release_request_blocks(req)
            req.release_logical_blocks()
            return
        if not manager.can_allocate(req):
            self.stats["prefetch_capacity_rejected"] += 1
            manager.release_request_blocks(req)
            req.release_logical_blocks()
            return
        req.control_prefetch = True
        req.control_prefetch_callback = self.controller.prefetch_completed
        state["request"] = req
        try:
            manager.allocate(req)
            connector.begin_load(req, boundary, hits)
        except OutOfBlocksError:
            if req.id in manager.tables:
                manager.release_request_blocks(req)
            self.stats["prefetch_capacity_rejected"] += 1
            req.release_logical_blocks()
            return
        self.stats["prefetch_started"] += 1
        self.stats["prefetch_bytes"] += size
        self.controller.record(
            "copy_started", owner=snapshot.owner, request_id=req.id, bytes=size
        )
        yield connector.loads[req.id]
        if req.cancel_requested:
            self.stats["prefetch_cancelled"] += 1
        else:
            self.stats["prefetch_completed"] += 1
        self.controller.record(
            "copy_finished",
            owner=snapshot.owner,
            request_id=req.id,
            cancelled=req.cancel_requested,
        )

    def resuming(self, req, *, admitted=False):
        owner = self.controller.owner(req)
        enhanced = getattr(
            self.controller,
            "queue_window_enabled",
            getattr(self.controller, "mode", "movement") == "enhanced_coop",
        )
        started = self.waiting.get(owner)
        if started is not None:
            observed = max(0, self.env.now - started)
            estimates = self.admitted_estimates if admitted else self.estimates
            estimates[owner] = 0.8 * estimates.get(owner, observed) + 0.2 * observed
            if admitted or not enhanced:
                self.waiting.pop(owner, None)
        state = self.jobs.get(owner)
        if (
            enhanced
            and not admitted
            and (state is None or state["snapshot"].compatible(req))
        ):
            return
        state = self.jobs.get(owner)
        if state is None:
            return
        if state["phase"] in ("moving", "sleeping", "idle"):
            if state["process"].is_alive:
                state["process"].interrupt()
        elif (
            state["phase"] == "prefetch"
            and state["request"] is not None
            and not state["snapshot"].compatible(req)
        ):
            state["request"].cancel_requested = True

    def blocked(self, scheduler, req):
        state = self.jobs.get(self.controller.owner(req))
        blocked = bool(
            state
            and state["phase"] == "prefetch"
            and not state["done"].triggered
            and state["snapshot"].connector.scheduler is scheduler
            and state["snapshot"].compatible(req)
        )
        if blocked:
            self.blocked_since.setdefault(req.id, self.env.now)
        elif req.id in self.blocked_since:
            self.stats[
                "foreground_wait_seconds"
            ] += self.env.now - self.blocked_since.pop(req.id)
        return blocked

    def release(self, owner):
        state = self.jobs.pop(owner, None)
        if state is not None:
            if state["phase"] == "prefetch" and state["request"] is not None:
                state["request"].cancel_requested = True
            elif state["process"].is_alive:
                state["process"].interrupt()
        self.snapshots.pop(owner, None)
        self.snapshot_revision += 1
        self.waiting.pop(owner, None)
        self.estimates.pop(owner, None)
        self.admitted_estimates.pop(owner, None)

    def close(self):
        self.closed = True
        for owner in tuple(self.jobs):
            self.release(owner)
