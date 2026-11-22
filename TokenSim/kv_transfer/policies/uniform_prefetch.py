"""Tool-pause copying of previously observed uniform prefixes."""

from dataclasses import dataclass
import simpy

from TokenSim.llm.llm_request import Request


@dataclass(frozen=True)
class UniformSnapshot:
    request_id: int
    owner: str
    connector: object
    prompt_tokens: int
    hashes: tuple
    cache_salt: str
    reuse_group: str
    chat_id: str
    session_key: object
    keys: tuple
    blocks: tuple
    content: tuple

    def compatible(self, req):
        return (
            req.cache_salt == self.cache_salt
            and tuple(req.hash_ids[: len(self.hashes)]) == self.hashes
        )


class UniformPrefetch:
    def __init__(self, controller):
        self.controller = controller
        self.env = controller.env
        self.snapshots = {}
        self.waiting = {}
        self.estimates = {}
        self.jobs = {}
        self.next_id = -2000000000000
        self.stats = dict(
            started=0,
            completed=0,
            cancelled=0,
            capacity_rejected=0,
            budget_bytes=0,
            copied_bytes=0,
            useful_bytes=0,
            wasted_bytes=0,
            benefiting_requests=0,
            issued_bytes=0,
            discarded_read_bytes=0,
        )
        self.pages = {}

    def capture(self, scheduler, req):
        if getattr(req, "parent_chat_id", None) is not None or not req.hash_ids:
            return
        connector = scheduler.connector
        keys = tuple(connector._keys_for_request(req))
        blocks = tuple(
            scheduler.block_manager.block_table.get_blocks(req.id)[: len(keys)]
        )
        if len(keys) != len(blocks):
            return
        gpu_keys = tuple(req.input_cache_keys[: len(keys)])
        self.snapshots[self.controller.owner(req)] = UniformSnapshot(
            req.id,
            self.controller.owner(req),
            connector,
            len(keys) * connector.block_size,
            tuple(req.hash_ids[: len(keys)]),
            req.cache_salt,
            req.reuse_group,
            req.chat_id,
            getattr(req, "session_key", None),
            keys,
            blocks,
            gpu_keys,
        )

    def tool_wait(self, req):
        owner = self.controller.owner(req)
        snapshot = self.snapshots.get(owner)
        if snapshot is None:
            return
        self.waiting[owner] = self.env.now
        if snapshot.connector.policy.reuse_threshold > 1:
            selected = [
                (key, block)
                for key, block, content in zip(
                    snapshot.keys, snapshot.blocks, snapshot.content
                )
                if block.cached and block.block_hash == content
            ]
            snapshot.connector.queue_save(
                [p[0] for p in selected],
                [p[1] for p in selected],
                declared_wait=True,
                request_id=snapshot.request_id,
            )
        if not snapshot.connector.policy.prefetch or owner not in self.estimates:
            return
        self.release(owner)
        state = dict(snapshot=snapshot, phase="sleeping", request=None)
        self.jobs[owner] = state
        state["process"] = self.env.process(self._copy(state))

    def _copy(self, state):
        snapshot = state["snapshot"]
        connector = snapshot.connector
        owner = snapshot.owner
        try:
            count = min(
                len(snapshot.keys),
                connector.policy.prefetch_budget_bytes // connector.block_bytes,
            )
            if not count:
                return
            store = connector.service.store
            cost = 0.0
            if store.offload is not None:
                from TokenSim.storage import IORequest

                cost = sum(
                    store.offload.backend.estimate_io(
                        IORequest("read", connector.block_bytes)
                    ).service_seconds
                    for key in snapshot.keys[:count]
                    if key.to_string() in store.objects
                    and store.objects[key.to_string()].tier != "memory"
                )
            delay = max(
                0.0,
                self.estimates[owner] - cost - connector.policy.prefetch_lead_seconds,
            )
            yield self.env.timeout(delay)
            req = Request(
                self.next_id,
                count * connector.block_size + 1,
                1,
                connector.block_size,
                hash_ids=list(snapshot.hashes[:count]),
                cache_salt=snapshot.cache_salt,
                reuse_group=snapshot.reuse_group,
                chat_id=snapshot.chat_id,
                record_timing=False,
            )
            self.next_id -= 1
            req.control_prefetch = True
            req.control_prefetch_callback = self.completed
            state["request"] = req
            manager = connector.manager
            plan = manager.plan_reuse(req)
            boundary, keys = connector.probe(req, plan)
            while (
                not keys
                and owner in self.waiting
                and not self.controller.closed
                and any(
                    key.to_string() in connector.state.pending_keys
                    for key in snapshot.keys[:count]
                )
            ):
                yield connector.changed
                plan = manager.plan_reuse(req)
                boundary, keys = connector.probe(req, plan)
            if not keys:
                req.release_logical_blocks()
                return
            if not manager.can_allocate(req):
                self.stats["capacity_rejected"] += 1
                req.release_logical_blocks()
                return
            manager.allocate(req)
            req.gpu_cached_context_tokens = min(plan.hit_tokens, req.context_len - 1)
            state["phase"] = "loading"
            self.stats["started"] += 1
            self.stats["budget_bytes"] += count * connector.block_bytes
            self.stats["issued_bytes"] += len(keys) * connector.block_bytes
            self.controller.record(
                "prefetch_started",
                owner=owner,
                request_id=req.id,
                bytes=len(keys) * connector.block_bytes,
            )
            connector.begin_load(req, boundary, keys)
            yield connector.loads[req.id]
            if req.cancel_requested:
                self.stats["discarded_read_bytes"] += len(keys) * connector.block_bytes
        except simpy.Interrupt:
            pass
        finally:
            req = state.get("request")
            if req is not None and not connector.manager.block_table.get_blocks(req.id):
                req.release_logical_blocks()
            if self.jobs.get(owner) is state:
                self.jobs.pop(owner, None)
            self.controller.notify()

    def completed(self, scheduler, req, blocks):
        connector = scheduler.connector
        size = len(blocks) * connector.block_bytes
        self.stats["completed"] += 1
        self.stats["copied_bytes"] += size
        self.controller.record("prefetch_finished", request_id=req.id, bytes=size)
        owner = self.controller.owner(req)
        state = self.jobs.get(owner)
        if state:
            state["phase"] = "ready"
        for block in blocks:
            key = (id(connector.manager), block.block_number, block.content_epoch)
            self.pages[key] = dict(
                block=block,
                content=block.block_hash,
                bytes=connector.block_bytes,
                owner=owner,
                used=False,
            )

    def resumed(self, scheduler, req):
        hit = (req.gpu_cached_context_tokens + req.block_size - 1) // req.block_size
        benefited = False
        for block in scheduler.block_manager.block_table.get_blocks(req.id)[:hit]:
            key = (id(scheduler.block_manager), block.block_number, block.content_epoch)
            record = self.pages.get(key)
            if record and not record["used"] and record["content"] == block.block_hash:
                record["used"] = True
                self.stats["useful_bytes"] += record["bytes"]
                benefited = True
        if benefited:
            self.stats["benefiting_requests"] += 1

    def client_resuming(self, req):
        owner = self.controller.owner(req)
        started = self.waiting.pop(owner, None)
        if started is not None:
            observed = self.env.now - started
            self.estimates[owner] = (
                0.8 * self.estimates.get(owner, observed) + 0.2 * observed
            )
        state = self.jobs.get(owner)
        if state and state["phase"] == "sleeping":
            self.release(owner)
        elif (
            state
            and state["phase"] == "loading"
            and not state["snapshot"].compatible(req)
        ):
            self.release(owner)

    def blocks(self, scheduler, req):
        state = self.jobs.get(self.controller.owner(req))
        return bool(
            state
            and state["phase"] == "loading"
            and state["snapshot"].connector.scheduler is scheduler
            and state["snapshot"].compatible(req)
        )

    def release(self, owner):
        state = self.jobs.get(owner)
        if state:
            if state["phase"] == "loading":
                state["request"].cancel_requested = True
                self.stats["cancelled"] += 1
            elif state["process"].is_alive:
                state["process"].interrupt()

    def finish_play(self, session_key):
        if session_key is None:
            return
        for owner, snapshot in list(self.snapshots.items()):
            if snapshot.session_key == session_key:
                self.release(owner)
                self.snapshots.pop(owner, None)
                self.waiting.pop(owner, None)
                self.estimates.pop(owner, None)
                for key, record in list(self.pages.items()):
                    if record["owner"] == owner:
                        self.pages.pop(key)

    def close(self):
        for owner in list(self.jobs):
            self.release(owner)

    @property
    def pending(self):
        return any(state["process"].is_alive for state in self.jobs.values())
