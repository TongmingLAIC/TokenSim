"""Queued restoration of known uniform prefix objects into pinned GPU pages."""

from TokenSim.llm.llm_request import RequestStatus


class UniformLoadMixin:
    def probe(self, req, local_plan):
        if self.config.kv_role == "kv_producer" or not self.options.enable_lookup:
            return local_plan.hit_tokens, ()
        keys = self._keys_for_request(req)
        start = min(len(keys), local_plan.hit_tokens // self.block_size)
        hit = self.service.store.probe(keys[start:], now=self.env.now)
        selected = list(hit.keys)
        if self._aligned:
            limit = max(0, (req.prefill_len - 1) // self.block_size - start)
            selected = selected[:limit]
        boundary = min(
            (start + len(selected)) * self.block_size, max(0, req.context_len - 1)
        )
        return boundary, tuple(selected)

    def begin_load(self, req, boundary, keys):
        offset = req.gpu_cached_context_tokens // self.block_size
        blocks = self.manager.block_table.get_blocks(req.id)[
            offset : offset + len(keys)
        ]
        if len(blocks) != len(keys):
            raise ValueError(
                "Uniform load destination does not cover its source prefix"
            )
        self.state.pin(keys)
        self.pin_gpu(blocks)
        req.external_cached_context_tokens = boundary - req.gpu_cached_context_tokens
        req.status = RequestStatus.WAITING_FOR_KV
        self.scheduler.loading[req.id] = req
        self.loads[req.id] = self.env.process(
            self._load(req, offset, tuple(keys), blocks, boundary)
        )

    def _load(self, req, offset, keys, blocks, boundary):
        issued = self.env.now
        nbytes = len(keys) * self.block_bytes
        control = getattr(req, "control_prefetch", False)
        store = self.service.store
        ready = False
        try:
            wire = self.wire_latency(nbytes)
            tiers = tuple(store.object_tier(key) for key in keys)

            def read():
                return wire + store.get(
                    list(keys),
                    None,
                    now=self.env.now,
                    initial_delay=wire,
                    stats=self.mooncake_stats,
                )

            yield self.env.process(
                self.state.transaction(
                    "read",
                    nbytes,
                    read,
                    prefetch=control,
                    tags=dict(request_id=req.id, worker_id=self.worker.id),
                )
            )
            restored = not req.cancel_requested and self.manager.block_table.get_blocks(
                req.id
            )
            if restored:
                target_keys = req.input_cache_keys[offset : offset + len(keys)]
                self.manager.kv_cache_manager.register_blocks(blocks, target_keys)
                if req.needs_recompute:
                    req.recompute_tokens = max(0, req.context_len - boundary)
                req.start_context_build(boundary)
                req.status = RequestStatus.WAITING
                if control:
                    req.control_prefetch_callback(self.scheduler, req, blocks)
                else:
                    self.scheduler.loaded_waiting.append(req)
                    ready = True
                if not control:
                    from_disk = sum(
                        min(
                            self.block_size,
                            max(
                                0,
                                req.external_cached_context_tokens
                                - i * self.block_size,
                            ),
                        )
                        for i, tier in enumerate(tiers)
                        if tier != "memory"
                    )
                    store.record_lookup(
                        store.offload.tier if from_disk else "memory",
                        stats=self.mooncake_stats,
                    )
                    self.mooncake_stats.mooncake_disk_hit_tokens += from_disk
                    self.mooncake_stats.mooncake_memory_hit_tokens += (
                        req.external_cached_context_tokens - from_disk
                    )
            self.record_transfer(
                "load", len(keys), nbytes, self.env.now - issued, blocking=not control
            )
            self.observation.read(req, issued, self.env.now, prefetch=control)
        finally:
            if not ready:
                self.manager.release_request_blocks(req)
                self.scheduler.loading.pop(req.id, None)
                self.on_request_released(req)
                if control:
                    req.release_logical_blocks()
            self.state.unpin(keys)
            self.unpin_gpu(blocks)
            self.loads.pop(req.id, None)
            self.notify()
