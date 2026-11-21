"""Observed reuse admission over native uniform object writes."""


class UniformSaveMixin:
    def on_new_request(self, req):
        self.state.observe(self._keys_for_request(req))
        self.mooncake_stats.cache_query_tokens += (
            len(self._keys_for_request(req)) * self.block_size
        )

    def on_tokens_computed(self, req):
        if self.config.kv_role == "kv_consumer" or req.cancel_requested:
            return
        if req.num_computed_tokens < req.prefill_len:
            return
        if (
            self.options.save_policy != "every_step"
            and req.id in self._saved_request_ids
        ):
            return
        self._saved_request_ids.add(req.id)
        keys = self._keys_for_request(req)
        blocks = self.manager.block_table.get_blocks(req.id)[: len(keys)]
        self.queue_save(keys, blocks, request_id=req.id)

    def queue_save(self, keys, blocks, *, declared_wait=False, request_id=None):
        if self.options.admission_policy == "never":
            return
        selected = []
        missing = set(self.service.store.missing_indices(list(keys)))
        for index, (key, block) in enumerate(zip(keys, blocks)):
            name = key.to_string()
            if self._aligned and (
                index not in missing or name in self.state.pending_keys
            ):
                continue
            if (
                not declared_wait
                and self.state.count(key) < self.policy.reuse_threshold
            ):
                self.policy_stats["reuse_rejected_bytes"] += self.block_bytes
                continue
            selected.append((key, block))
        batch = max(1, self.policy.save_batch_bytes // self.block_bytes)
        for offset in range(0, len(selected), batch):
            items = selected[offset : offset + batch]
            size = len(items) * self.block_bytes
            if not self.state.queue.reserve_write(size):
                continue
            keys = tuple(k for k, _ in items)
            blocks = tuple(b for _, b in items)
            for key in keys:
                self.state.pending_keys.add(key.to_string())
            self.pin_gpu(blocks)
            self.write_jobs += 1
            self.env.process(self._save(keys, blocks, request_id))

    def _save(self, keys, blocks, request_id):
        issued = self.env.now
        size = len(keys) * self.block_bytes
        try:
            wire = self.wire_latency(size)

            def write():
                timing = self.service.store.put_with_timing(
                    list(keys),
                    now=self.env.now,
                    initial_delay=wire,
                    stats=self.mooncake_stats,
                )
                return wire + timing.accounting_latency

            yield self.env.process(
                self.state.transaction(
                    "write",
                    size,
                    write,
                    tags=dict(request_id=request_id, worker_id=self.worker.id),
                )
            )
            self.record_transfer(
                "save", len(keys), size, self.env.now - issued, blocking=False
            )
        finally:
            for key in keys:
                self.state.pending_keys.discard(key.to_string())
            self.unpin_gpu(blocks)
            self.write_jobs -= 1
            self.notify()

    def wait_for_save(self):
        return 0.0
