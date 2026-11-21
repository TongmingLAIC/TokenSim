"""Optional online KV control over native allocation, scheduling and transfers."""

import json
from pathlib import Path

from TokenSim.block.priority_queue import PriorityFreeMap
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer.retention import RetentionRegistry


class NativeCacheController:
    def __init__(self, engine):
        self.engine, self.env = engine, engine.env
        self.mode = engine.config.cache_policy
        self.queue_window_enabled = self.mode == "enhanced_coop" and getattr(
            engine.config, "cache_queue_window_enabled", True
        )
        self.child_write_admission = getattr(
            engine.config, "cache_child_write_admission", True
        )
        self.child_reclamation = getattr(engine.config, "cache_child_reclamation", True)
        self.retain_enabled = self.mode in (
            "retain",
            "retain_movement",
            "all",
            "forced_movement",
            "enhanced_coop",
        )
        self.movement_enabled = self.mode in (
            "movement",
            "retain_movement",
            "all",
            "forced_movement",
            "enhanced_coop",
        )
        self.quota_enabled = self.mode in (
            "quota",
            "all",
            "forced_movement",
            "enhanced_coop",
        )
        self.priority = engine.config.cache_retention_priority
        self.ttl = engine.config.cache_retention_seconds
        self.registry = RetentionRegistry(lambda: self.env.now)
        self.namespaces = {}
        self.revision = 0
        self.play_owners = {}
        self.event_path = None
        self.closed = False
        self.movement = None
        if self.movement_enabled:
            from TokenSim.kv_transfer.movement import MovementRuntime

            self.movement = MovementRuntime(self)
        self.stats = dict(completions=0, cancellations=0, retained_blocks=0)
        self.admitted_at = {}
        self.service_estimate = 0.0
        if engine.cluster_config.af.enabled:
            raise ConfigurationError(
                "Cache-control experiments currently require ordinary serving workers"
            )
        for worker in engine.workers:
            scheduler = getattr(worker, "scheduler", None)
            manager = getattr(scheduler, "block_manager", None)
            if manager is None:
                continue
            scheduler.cache_controller = self
            namespace = f"worker:{worker.id}"
            self.namespaces[id(manager)] = namespace
            if self.movement_enabled and (
                not manager.grouped_cache or not scheduler.connector.async_offload
            ):
                raise ConfigurationError(
                    "Tool-wait movement requires native grouped asynchronous storage"
                )
            if not self.retain_enabled:
                continue
            if manager.grouped_cache:
                provider = lambda slot, m=manager, n=namespace: self.registry.priority(
                    n, m.blocks[slot]
                )
                queue = PriorityFreeMap(provider, lambda: self.env.now)
                queue.update(manager._free_cached)
                manager._free_cached = queue
                self.registry.bind(namespace, queue.refresh)
            else:
                pool = manager.gpu_allocator.pool
                provider = lambda slot, p=pool, n=namespace: self.registry.priority(
                    n, p._materialized[slot]
                )
                queue = pool.free_blocks
                queue.enable_priority(provider, lambda: self.env.now)
                self.registry.bind(namespace, queue.refresh_priority)

    @staticmethod
    def owner(req):
        return req.chat_id or f"request:{req.id}"

    def resumed(self, req):
        if self.retain_enabled:
            self.registry.release(self.owner(req))

    def defer_admission(self, scheduler, req):
        if self.movement is not None and self.movement.blocked(scheduler, req):
            return True
        if not self.quota_enabled:
            return False
        from TokenSim.kv_transfer.quota import should_defer

        blocked = should_defer(
            scheduler,
            req,
            now=self.env.now,
            fraction=self.engine.config.cache_quota_fraction,
            max_wait_seconds=self.engine.config.cache_quota_max_wait_seconds,
        )
        if blocked:
            scheduler.stats["kv_quota_deferred"] = (
                scheduler.stats.get("kv_quota_deferred", 0) + 1
            )
        return blocked

    def reorder_waiting(self, scheduler):
        if not self.quota_enabled or len(scheduler.waiting) < 2:
            return
        from TokenSim.kv_transfer.quota import AdmissionPressure, should_defer

        pressure = AdmissionPressure(scheduler)

        def key(req):
            plan = scheduler.block_manager.kv_cache_manager.plan_reuse(req)
            deferred = should_defer(
                scheduler,
                req,
                now=self.env.now,
                fraction=self.engine.config.cache_quota_fraction,
                max_wait_seconds=self.engine.config.cache_quota_max_wait_seconds,
                pressure=pressure,
                plan=plan,
            )
            hit = plan.hit_tokens
            age = max(0, self.env.now - req.arrival_at)
            parent = getattr(req, "parent_chat_id", None) is None
            score = hit + age / 30 * 2048 + int(parent) * 1024
            return deferred, -score, req.arrival_at

        scheduler.waiting.sort(key=key)

    def admitted(self, scheduler, req):
        self.admitted_at.setdefault(req.id, self.env.now)
        if self.movement is not None and self.queue_window_enabled:
            self.movement.resuming(req, admitted=True)

    def queue_estimate(self, scheduler):
        parallel = max(1, len(scheduler.running))
        return self.service_estimate * len(scheduler.waiting) / parallel

    def skip_write(self, manager, block, req):
        from TokenSim.kv_transfer.child_policy import skip_write

        skipped = skip_write(self, manager, block, req)
        if skipped:
            self.stats["child_write_bypassed_pages"] = (
                self.stats.get("child_write_bypassed_pages", 0) + 1
            )
        return skipped

    def after_free(self, scheduler, req, block_ids):
        from TokenSim.kv_transfer.child_policy import reclaim_child

        reclaim_child(self, scheduler, req, block_ids)

    def completed(self, scheduler, req):
        began = self.admitted_at.pop(req.id, None)
        if began is not None and not req.cancel_requested:
            service = max(0, self.env.now - began)
            self.service_estimate = (
                0.8 * self.service_estimate + 0.2 * service
                if self.service_estimate
                else service
            )
        self.stats["completions"] += 1
        scheduler.stats["kv_control_completions"] = (
            scheduler.stats.get("kv_control_completions", 0) + 1
        )
        if req.cancel_requested:
            return
        if getattr(req, "parent_chat_id", None) is not None:
            return
        if self.movement is not None:
            self.movement.capture(scheduler, req)
        source = getattr(req, "source", None)
        if source is not None and hasattr(source, "play_id"):
            self.play_owners.setdefault(
                (source.trace_id, source.play_id, source.lane), set()
            ).add(self.owner(req))
        if not self.retain_enabled:
            return
        self.prefetch_completed(scheduler, req)

    def prefetch_completed(self, scheduler, req):
        if not self.retain_enabled or self.closed:
            return
        manager = scheduler.block_manager
        # Normal step publication has already tagged complete prompt pages.
        blocks = [
            b
            for b in manager.block_table.get_blocks(req.id)
            if b.block_hash is not None
        ]
        self.revision += 1
        self.registry.declare(
            self.namespaces[id(manager)],
            blocks,
            owner=self.owner(req),
            revision=self.revision,
            priority=self.priority,
            ttl=self.ttl,
        )
        self.stats["retained_blocks"] += len(blocks)
        self.record(
            "set_priority",
            owner=self.owner(req),
            pages=len(blocks),
            priority=self.priority,
            ttl_seconds=self.ttl,
        )

    def client_resuming(self, req):
        if self.movement is not None and req.measurement_phase == "profile":
            self.movement.resuming(req)

    def tool_wait(self, req):
        if self.movement is not None and req.measurement_phase == "profile":
            self.movement.tool_wait(req)

    def play_finished(self, identity):
        for owner in self.play_owners.pop(identity, ()):
            self.registry.release(owner)
            if self.movement is not None:
                self.movement.release(owner)

    def cancelled(self, req):
        self.admitted_at.pop(req.id, None)
        self.stats["cancellations"] += 1
        self.registry.release(self.owner(req))
        if self.movement is not None:
            self.movement.release(self.owner(req))

    def configure_log(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.event_path = directory / "cache_control.jsonl"
        self.event_path.write_text("")
        self.record("controller_enabled", mode=self.mode)

    def record(self, kind, **values):
        if self.event_path is not None:
            with self.event_path.open("a") as stream:
                stream.write(
                    json.dumps(dict(time_s=self.env.now, kind=kind, **values)) + "\n"
                )

    @property
    def pending(self):
        return self.movement is not None and bool(self.movement.active)

    @property
    def changed(self):
        return self.movement.changed

    def close(self):
        self.closed = True
        if self.movement is not None:
            self.movement.close()
        self.registry.reset()

    def write_summary(self, directory):
        Path(directory, "cache_control_summary.json").write_text(
            json.dumps(self.describe(), indent=2) + "\n"
        )

    def describe(self):
        return dict(
            mode=self.mode,
            priority=self.priority,
            ttl_seconds=self.ttl,
            **self.stats,
            movement=self.movement.stats if self.movement else None,
            quota_fraction=self.engine.config.cache_quota_fraction,
            quota_max_wait_seconds=self.engine.config.cache_quota_max_wait_seconds,
            retention_semantics="Evictable priority/TTL claims; transfer reference counts are unchanged",
        )
