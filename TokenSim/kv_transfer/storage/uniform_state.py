"""Shared queue ownership and byte observations over the native object store."""

from collections import Counter, OrderedDict
import hashlib

from TokenSim.errors import SimulationStateError
from TokenSim.kv_transfer.storage.queue import StorageQueue


class UniformStoreState:
    def __init__(self, env, service, policy):
        self.env = env
        self.service = service
        self.policy = policy
        self.queue = StorageQueue(env, policy)
        self.pending_keys = set()
        self.read_pins = Counter()
        self.frequency = OrderedDict()
        self.latest_program = {}
        self.unread_program_bytes = 0
        self.program_bytes = 0
        self.read_bytes = 0
        self.media_context = None
        media = service.store.offload
        if media is not None:
            media.io_observers.append(self._observe_media)

    def _observe_media(self, kind, key, operation):
        if self.media_context is not None:
            self.media_context[kind].append((key, operation.bytes))
        for victim in operation.evicted_keys:
            self.latest_program.pop(victim, None)

    def observe(self, keys):
        for key in keys:
            digest = hashlib.blake2b(key.to_string().encode(), digest_size=16).digest()
            count, seen = self.frequency.get(digest, (0, self.env.now))
            if self.env.now - seen > self.policy.history_seconds:
                count = 0
            self.frequency[digest] = (count + 1, self.env.now)
            self.frequency.move_to_end(digest)
        while len(self.frequency) > self.policy.history_entries:
            self.frequency.popitem(last=False)

    def count(self, key):
        digest = hashlib.blake2b(key.to_string().encode(), digest_size=16).digest()
        count, seen = self.frequency.get(digest, (0, self.env.now))
        return count if self.env.now - seen <= self.policy.history_seconds else 0

    def pin(self, keys):
        store = self.service.store
        for key in keys:
            name = key.to_string()
            obj = store.objects[name]
            self.read_pins[name] += 1
            obj.in_use_until = float("inf")
            if store.offload is not None and obj.tier != "memory":
                store.offload.pin(name, float("inf"))

    def unpin(self, keys):
        store = self.service.store
        for key in keys:
            name = key.to_string()
            self.read_pins[name] -= 1
            if not self.read_pins[name]:
                del self.read_pins[name]
                if name in store.objects:
                    store.objects[name].in_use_until = self.env.now
                if store.offload is not None:
                    store.offload.release_pin(name)

    def transaction(self, kind, nbytes, operation, *, prefetch=False, tags=None):
        context = dict(write=[], read=[])

        def execute():
            self.media_context = context
            try:
                return operation()
            finally:
                self.media_context = None

        def completed():
            for key, size in context["write"]:
                self.program_bytes += size
                self.unread_program_bytes += size
                self.latest_program[key] = size
            for key, size in context["read"]:
                self.read_bytes += size
                self.unread_program_bytes -= self.latest_program.pop(key, 0)

        yield self.env.process(
            self.queue.execute(
                kind,
                nbytes,
                execute,
                prefetch=prefetch,
                completed=completed,
                context=tags,
            )
        )

    def describe(self):
        return dict(
            **self.queue.describe(),
            pinned_store_objects=len(self.read_pins),
            pending_write_objects=len(self.pending_keys),
            nand_program_bytes=self.program_bytes,
            nand_read_bytes=self.read_bytes,
            nand_program_bytes_without_later_read=self.unread_program_bytes,
            media_reuse_definition="A completed NAND read of the latest programmed object, including prefetched reads",
        )


class UniformStoreView:
    def __init__(self, state):
        self.state = state

    @property
    def used_bytes(self):
        return self.state.service.store.memory_used_bytes

    def describe(self):
        store = self.state.service.store
        far = store.offload
        return dict(
            scope="shared",
            memory_used_bytes=store.memory_used_bytes,
            memory_capacity_bytes=store.memory_capacity_bytes,
            offload_used_bytes=far.used_blocks * store.block_bytes if far else 0,
            offload_capacity_bytes=far.backend.capacity_bytes if far else 0,
            **self.state.describe(),
        )

    def validate(self):
        store = self.state.service.store
        memory = sum(
            obj.bytes for obj in store.objects.values() if obj.tier == "memory"
        )
        if memory != store.memory_used_bytes or memory > store.memory_capacity_bytes:
            raise SimulationStateError(
                "Uniform store memory bytes do not conserve capacity"
            )
        if (
            store.offload is not None
            and store.offload.used_blocks > store.offload.capacity_blocks
        ):
            raise SimulationStateError("Uniform offload capacity exceeded")
        if (
            self.state.read_pins
            or self.state.queue.backlog_bytes
            or self.state.pending_keys
        ):
            raise SimulationStateError("Uniform storage still owns unfinished I/O")
