"""Two finite stores with visibility and reference ownership per physical copy."""

from TokenSim.kv_transfer.storage.native import NativePageStore
from TokenSim.config.validation import integer
from TokenSim.kv_transfer.storage.tiered_service import TierPage


class TieredStore:
    def __init__(self, connector, namespace):
        self.connector, self.namespace = connector, namespace
        options = connector.options
        stride = connector.manager.layout.physical_slot_bytes
        self.local = NativePageStore(
            options.l2_bytes_per_rank - options.staging_bytes_per_rank,
            allocation_unit=stride,
            scope=f"tiered:l2:{connector.worker.id}",
            before_evict=self.before_evict,
        )
        self.remote = connector.service.store
        self.payload_bytes = {}
        self.one_shot = set()
        self.failed_admission_state = None
        self.stats = dict(
            l2_read_pages=0,
            l3_read_pages=0,
            promotion_pages=0,
            refused_l2_writes=0,
            selective_discard_pages=0,
        )

    @property
    def used_bytes(self):
        return self.local.used_bytes

    @property
    def capacity_bytes(self):
        return self.local.capacity_bytes

    def remote_key(self, key):
        return self.namespace, key

    def lookup(self, key):
        local = self.local.lookup(key)
        if local is not None:
            return TierPage(key, local, self.local, 2)
        remote = self.remote.lookup(self.remote_key(key))
        return TierPage(key, remote, self.remote, 3) if remote is not None else None

    def reserve_write(self, key, size):
        integer(size, "payload_bytes", minimum=1)
        if key in self.local.pages:
            return None
        state = (
            self.local.mutation_version,
            self.remote.mutation_version,
            self.local.capacity_bytes,
            self.remote.capacity_bytes,
        )
        if self.failed_admission_state == state:
            self.stats["refused_l2_writes"] += 1
            return None
        local = self.local.reserve_write(key, size)
        if local is None:
            self.failed_admission_state = (
                self.local.mutation_version,
                self.remote.mutation_version,
                self.local.capacity_bytes,
                self.remote.capacity_bytes,
            )
            self.stats["refused_l2_writes"] += int(key not in self.local.pages)
            return None
        self.payload_bytes[key] = size
        return TierPage(key, local, self.local, 2)

    def finish_write(self, view):
        view.backing.finish_write(view.page)

    def abort_write(self, view):
        view.backing.abort_write(view.page)
        if view.tier == 2:
            self.payload_bytes.pop(view.key, None)
            self.one_shot.discard(view.key)

    def acquire(self, view):
        view.backing.acquire(view.page)
        self.stats[f"l{view.tier}_read_pages"] += 1

    def release(self, view):
        view.backing.release(view.page)

    def has_remote(self, key):
        page = self.remote.pages.get(self.remote_key(key))
        return page is not None and page.ready

    def before_evict(self, page):
        policy = self.connector.options.write_policy
        if self.has_remote(page.key):
            self.payload_bytes.pop(page.key, None)
            self.one_shot.discard(page.key)
            return True
        if policy in ("write_back", "selective_write_back"):
            if policy == "selective_write_back" and page.key in self.one_shot:
                self.stats["selective_discard_pages"] += 1
            elif self.connector.enqueue_flush(
                page, evict=self.connector.options.watermark_eviction
            ):
                return False
            else:
                # Admission failure preserves the local copy; another clean
                # victim may still permit the incoming write.
                return False
        self.payload_bytes.pop(page.key, None)
        self.one_shot.discard(page.key)
        return True

    def pressure_flush(self):
        options = self.connector.options
        if options.write_policy not in ("write_back", "selective_write_back"):
            return
        if self.local.used_bytes < options.high_watermark * self.local.capacity_bytes:
            return
        evict = options.watermark_eviction
        budget = (
            self.local.used_bytes
            - options.low_watermark * self.local.capacity_bytes
            - self.connector.pending_eviction_bytes
            if evict
            else (options.high_watermark - options.low_watermark)
            * self.local.capacity_bytes
        )
        if budget <= 0:
            return
        for key in list(self.local.lru):
            page = self.local.pages[key]
            if not page.ready or page.references:
                continue
            one_shot = (
                options.write_policy == "selective_write_back" and key in self.one_shot
            )
            size = page.bytes
            if evict and (one_shot or self.has_remote(key)):
                if self.drop_local(page):
                    budget -= size
            elif not one_shot and self.connector.enqueue_flush(page, evict=evict):
                budget -= size
            if budget <= 0:
                break

    def drop_local(self, page):
        if not self.local.discard(page):
            return False
        self.payload_bytes.pop(page.key, None)
        self.one_shot.discard(page.key)
        return True

    def promote(self, view):
        if not self.connector.options.promote_reads:
            return
        local = self.local.reserve_write(
            view.key, self.payload_bytes.get(view.key, view.bytes)
        )
        if local is not None:
            self.payload_bytes[view.key] = view.bytes
            self.local.finish_write(local)
            self.stats["promotion_pages"] += 1

    def missing_bytes(self):
        return sum(
            self.payload_bytes.get(k, p.bytes)
            for k, p in self.local.pages.items()
            if p.ready and not self.has_remote(k)
        )

    def validate(self):
        self.local.validate()
        self.remote.validate()
        return True

    def describe(self):
        return dict(
            l2=self.local.describe(),
            l3=self.remote.describe(),
            l3_missing_payload_bytes=self.missing_bytes(),
            **self.stats,
        )
