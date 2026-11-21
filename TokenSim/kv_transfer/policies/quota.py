"""Session admission guidance based on physical references and declared growth."""

from math import ceil


def session_id(req):
    source = getattr(req, "source", None)
    if source is not None and all(
        hasattr(source, key) for key in ("trace_id", "play_id", "lane")
    ):
        return f"play:{source.trace_id}:{source.play_id}:{source.lane}"
    return req.cache_salt or req.chat_id or f"request:{req.id}"


def maximum_slots(scheduler, req):
    manager = scheduler.block_manager
    limit = scheduler.cache_config.model_spec.max_position_embeddings
    length = min(limit, req.prefill_len + req.decode_len)
    if manager.grouped_cache:
        layout = manager.layout
        if not hasattr(layout, "groups"):
            return layout.max_request_slots(length)
        # The layout's max-in-flight envelope covers whole prefill batches.
        # Reserve per-request decode growth, not that global scratch envelope.
        return sum(
            group.max_request_slots(length, 1 + layout.speculative_tokens)
            for group in layout.groups
        )
    return ceil(length / manager.block_size)


def hit_blocks(manager, plan):
    return (
        [block for _, pairs in plan.group_hits for _, block in pairs]
        if manager.grouped_cache
        else plan.hit_blocks
    )


class AdmissionPressure:
    """One immutable resource view for a waiting-queue scoring pass."""

    def __init__(self, scheduler, *, excluded=None):
        self.scheduler = scheduler
        manager = scheduler.block_manager
        active = {r.id: r for r in scheduler.running}
        active.update(getattr(scheduler, "loading", {}))
        active.update({r.id: r for r in getattr(scheduler, "loaded_waiting", ())})
        self.active_ids = set(active)
        owned = {r.id: manager.block_table.get_blocks(r.id) for r in active.values()}
        growth = {
            r.id: max(0, maximum_slots(scheduler, r) - len(owned[r.id]))
            for r in active.values()
        }
        global_ids = self.active_ids - {excluded}
        self.has_active = bool(global_ids)
        self.blocks = {b.block_number for ident in global_ids for b in owned[ident]}
        self.growth = sum(growth[ident] for ident in global_ids)
        self.sessions = {}
        for req in scheduler.running:
            blocks, future = self.sessions.setdefault(session_id(req), (set(), []))
            blocks.update(b.block_number for b in owned[req.id])
            future.append(growth[req.id])
        self.sessions = {
            key: (blocks, sum(future))
            for key, (blocks, future) in self.sessions.items()
        }

    def exceeds(self, req, hit_ids, watermark=0.92):
        if not self.has_active:
            return False
        incoming = max(
            0, maximum_slots(self.scheduler, req) - len(hit_ids & self.blocks)
        )
        return (
            len(self.blocks) + self.growth + incoming
            > watermark * self.scheduler.block_manager.num_total_gpu_blocks
        )

    def deferred(self, req, plan, *, now, fraction, max_wait_seconds):
        manager = self.scheduler.block_manager
        hits = {b.block_number for b in hit_blocks(manager, plan)}
        if self.exceeds(req, hits):
            return True
        if not self.scheduler.running or now - req.arrival_at >= max_wait_seconds:
            return False
        state = self.sessions.get(session_id(req))
        if state is None:
            return False
        blocks, growth = state
        demand = (
            len(blocks)
            + growth
            + max(0, maximum_slots(self.scheduler, req) - len(hits & blocks))
        )
        return demand > fraction * manager.num_total_gpu_blocks


def growth_exceeds_capacity(scheduler, req, plan, watermark=0.92):
    pressure = AdmissionPressure(scheduler, excluded=req.id)
    hits = {b.block_number for b in hit_blocks(scheduler.block_manager, plan)}
    return pressure.exceeds(req, hits, watermark)


def should_defer(
    scheduler, req, *, now, fraction, max_wait_seconds, pressure=None, plan=None
):
    if pressure is None or req.id in pressure.active_ids:
        pressure = AdmissionPressure(scheduler, excluded=req.id)
    if plan is None:
        plan = scheduler.block_manager.kv_cache_manager.plan_reuse(req)
    return pressure.deferred(
        req, plan, now=now, fraction=fraction, max_wait_seconds=max_wait_seconds
    )
