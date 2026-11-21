"""Private destinations for physical grouped KV handoff between serving domains."""

from TokenSim.errors import SimulationStateError


class GroupedHandoffMixin:
    def allocate_handoff_destinations(self, req, pages, local_hits=()):
        """Allocate request-owned pages without rewriting its prefill observations."""
        if req.id in self.tables:
            raise SimulationStateError("Handoff destination already owns this request")
        self.tables[req.id] = {spec.group_id: {} for spec in self.groups}
        self.requests[req.id] = req
        for gid, pairs in local_hits:
            for index, block in pairs:
                self._acquire(block)
                self.tables[req.id][gid][index] = block
        self.cached_progress[req.id] = req.num_computed_tokens // self.block_size
        self._spans[req.id] = self._window_spans(self.tables[req.id])
        return self.allocate_external_destinations(req, pages)

    def finish_handoff(self, req, destinations):
        """Publish only identities already proven complete by the source."""
        for _, _, source, block in destinations:
            if source.cached and source.block_hash is not None:
                key = source.block_hash
                existing = self.cache.get(key)
                block.block_hash, block.cached, block.is_full = key, True, True
                if existing is None:
                    self.cache[key] = block
                elif existing is not block:
                    self.duplicates.setdefault(key, {})[block.block_number] = block
        self._cache_version += 1
        self._committed.pop(req.id, None)
