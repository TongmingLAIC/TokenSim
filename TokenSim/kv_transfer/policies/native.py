"""Optional online KV control over native allocation, scheduling and transfers."""

from functools import partial

from TokenSim.config.cache_policy import PRESETS
from TokenSim.errors import ConfigurationError
from TokenSim.reporting.cache_control import CacheControlLog, write_summary
from TokenSim.kv_transfer.policies.retention import RetentionRegistry


class NativeCacheController:
    def __init__(self, engine):
        self.engine, self.env = engine, engine.env
        self.mode = engine.config.cache_policy
        self.preset = PRESETS[self.mode]
        self.queue_window_enabled = self.preset.cooperative and getattr(
            engine.config, "cache_queue_window_enabled", True
        )
        self.child_write_admission = self.preset.cooperative and getattr(
            engine.config, "cache_child_write_admission", True
        )
        self.child_reclamation = self.preset.cooperative and getattr(
            engine.config, "cache_child_reclamation", True
        )
        self.priority = engine.config.cache_retention_priority
        self.ttl = engine.config.cache_retention_seconds
        self.registry = RetentionRegistry(lambda: self.env.now)
        self.namespaces = {}
        self.revision = 0
        self.play_owners = {}
        self._log = CacheControlLog(self.env, self.mode)
        self.closed = False
        self.movement = None
        if self.preset.movement:
            from TokenSim.kv_transfer.policies.movement import MovementRuntime

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
            if self.preset.movement and (
                not manager.grouped_cache or not scheduler.connector.async_offload
            ):
                raise ConfigurationError(
                    "Tool-wait movement requires native grouped asynchronous storage"
                )
            if not self.preset.retain:
                continue
            refresh = manager.enable_retention(
                partial(self.registry.priority, namespace), lambda: self.env.now
            )
            self.registry.bind(namespace, refresh)

    @staticmethod
    def owner(req):
        return req.chat_id or f"request:{req.id}"

    def resumed(self, req):
        if self.preset.retain:
            self.registry.release(self.owner(req))

    def defer_admission(self, scheduler, req):
        if self.movement is not None and self.movement.blocked(scheduler, req):
            return True
        if not self.preset.quota:
            return False
        from TokenSim.kv_transfer.policies.quota import should_defer

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
        if not self.preset.quota or len(scheduler.waiting) < 2:
            return
        from TokenSim.kv_transfer.policies.quota import AdmissionPressure, should_defer

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
            score = self.preset.waiting_score(hit_tokens=hit, age=age, root=parent)
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
        from TokenSim.kv_transfer.policies.child import skip_write

        skipped = skip_write(self, manager, block, req)
        if skipped:
            self.stats["child_write_bypassed_pages"] = (
                self.stats.get("child_write_bypassed_pages", 0) + 1
            )
        return skipped

    def after_free(self, scheduler, req, block_ids):
        from TokenSim.kv_transfer.policies.child import reclaim_child

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
        if getattr(req, "session_key", None) is not None:
            self.play_owners.setdefault(req.session_key, set()).add(self.owner(req))
        if not self.preset.retain:
            return
        self.prefetch_completed(scheduler, req)

    def prefetch_completed(self, scheduler, req):
        if not self.preset.retain or self.closed:
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
        if self.movement is not None and req.is_measured:
            self.movement.resuming(req)

    def tool_wait(self, req):
        if self.movement is not None and req.is_measured:
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
        self._log.configure_log(directory)

    def record(self, kind, **values):
        self._log.record(kind, **values)

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
        write_summary(directory, self.describe())

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
