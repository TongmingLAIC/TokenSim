"""Packet completion tails, drop-tail retry assumptions and finite receive slots.

All parameters describe synthetic scenarios; ordinary resource schedulers still
price wire contention. A packet flight holds no endpoint serialization budget.
"""

from collections import Counter
from dataclasses import replace
import hashlib
import math
import random
import simpy


def uniform(seed, identity):
    digest = hashlib.sha256(repr((seed, identity)).encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big")).random()


def packet_tail(config, identity, payload, request_ids):
    """Maximum of independent packet tails, sampled by per-request CDF inversion."""
    if not config.packet_tail_probability or not config.tail_scale_us or not payload:
        return 0.0
    count = max(1, math.ceil(payload / config.packet_bytes))
    ids = sorted(set(request_ids))
    # Distribute a fixed number of packets, preserving total payload packet count.
    tail = 0.0
    for i, request_id in enumerate(ids):
        packets = count // len(ids) + int(i < count % len(ids))
        if not packets:
            continue
        u = uniform(config.seed, (identity, request_id))
        survival = -math.expm1(math.log(max(u, 1e-300)) / packets)
        p = config.packet_tail_probability
        if survival < p:
            delay = config.tail_scale_us * (
                (p / max(survival, 1e-300)) ** (1 / config.tail_shape) - 1
            )
            tail = max(tail, delay / 1e6)
    return tail


class ReceiverQueue:
    def __init__(self, env, capacity, ordered):
        self.env, self.ordered = env, ordered
        self.slots = simpy.Resource(env, capacity=capacity)
        self.entries = {}
        self.order = []
        self.changed = env.event()
        self.max_occupied = 0
        self.max_ready = 0
        self.overtakes = 0
        self.history = []

    def _reserve(self, entry):
        try:
            yield entry["reservation"]
        except simpy.Interrupt:
            return
        if entry.get("cancelled"):
            return
        self.max_occupied = max(self.max_occupied, self.slots.count)
        self.history.append(
            dict(
                time=self.env.now,
                action="reserve",
                key=entry["key"],
                occupied=self.slots.count,
            )
        )
        entry["admitted"].succeed()

    def enter(self, key):
        if key not in self.entries:
            entry = dict(
                reservation=self.slots.request(),
                admitted=self.env.event(),
                arrived=False,
                key=key,
            )
            self.entries[key] = entry
            self.order.append(key)
            entry["process"] = self.env.process(self._reserve(entry))
        yield self.entries[key]["admitted"]

    def arrived(self, key):
        self.entries[key]["arrived"] = True
        self.max_ready = max(
            self.max_ready, sum(e["arrived"] for e in self.entries.values())
        )
        if self.order[0] != key:
            self.overtakes += 1
        while self.ordered and self.order[0] != key:
            yield self.changed

    def release(self, key):
        entry = self.entries.pop(key, None)
        if entry is None:
            return
        entry["cancelled"] = True
        process = entry["process"]
        if process.is_alive and self.env.active_process is not process:
            process.interrupt("receiver reservation cancelled")
        reservation = entry["reservation"]
        if reservation.triggered:
            self.slots.release(reservation)
        else:
            reservation.cancel()
        self.history.append(
            dict(
                time=self.env.now, action="release", key=key, occupied=self.slots.count
            )
        )
        self.order.remove(key)
        if not self.changed.triggered:
            self.changed.succeed()
        self.changed = self.env.event()


class OrderedArrivalStream:
    """Release a received message as soon as earlier posted messages arrive."""

    def __init__(self, env, ordered):
        self.env, self.ordered = env, ordered
        self.order = []
        self.changed = env.event()
        self.overtakes = 0
        self.next_post_index = 0

    def register(self, key):
        self.order.append(key)
        index = self.next_post_index
        self.next_post_index += 1
        return index

    def arrived(self, key):
        blockers = []
        if self.order[0] != key:
            self.overtakes += 1
        while self.ordered and self.order[0] != key:
            blockers.append(self.order[0])
            yield self.changed
        self.order.remove(key)
        self._wake()
        return tuple(blockers)

    def _wake(self):
        if not self.changed.triggered:
            self.changed.succeed()
        self.changed = self.env.event()

    def cancel_iteration(self, iteration):
        self.order[:] = [key for key in self.order if key[0] != iteration.iteration_id]
        self._wake()


class AFTransportEffects:
    def __init__(self, executor):
        self.executor, self.env = executor, executor.env
        self.config = executor.cluster_config.af.transport_effects
        self.receiver = ReceiverQueue(self.env, self.config.receive_cohorts, False)
        self.streams = {}
        self.consumers = {}
        self.inflight_bytes = Counter()
        self.records = []
        self.peak_inflight_bytes = 0

    def register_layer(self, block, layer, dag):
        if not self.config.enabled:
            return
        f_domain = self.executor.cluster_config.af_domains[1].name
        consumers = {
            node.node_id
            for node in dag.nodes
            if node.kind == "compute" and node.source_domain == f_domain
        }
        if consumers:
            self.consumers[(block.iteration_id, block.block_id, layer)] = consumers

    def complete_compute(self, block, layer, node_id):
        key = (block.iteration_id, block.block_id, layer)
        pending = self.consumers.get(key)
        if pending is not None:
            pending.discard(node_id)
            if not pending:
                self.release_layer(block, layer)

    def release_layer(self, block, layer):
        key = (block.iteration_id, block.block_id, layer)
        self.consumers.pop(key, None)
        self.receiver.release(key)

    def cancel_iteration(self, iteration):
        for stream in self.streams.values():
            stream.cancel_iteration(iteration)
        for key in list(self.receiver.entries):
            if key[0] == iteration.iteration_id:
                self.consumers.pop(key, None)
                self.receiver.release(key)

    def run(self, block, layer, step, estimate, wire):
        cfg = self.config
        if not cfg.enabled or step.direction not in {"a_to_f", "f_to_a"}:
            return (yield from wire())
        ready = self.env.now
        key = (block.iteration_id, block.block_id, layer)
        stream = None
        stream_id = None
        post_index = None
        blockers = ()
        flow = (block.iteration_id, block.block_id, layer, step.node_id)
        if step.direction == "a_to_f":
            stream_id = (
                step.network,
                (
                    step.physical_workers[-1]
                    if hasattr(step, "physical_workers")
                    else step.target_endpoint
                ),
            )
            stream = self.streams.setdefault(
                stream_id, OrderedArrivalStream(self.env, cfg.ordered_receive)
            )
            yield from self.receiver.enter(key)
            post_index = stream.register(flow)
        admitted = self.env.now
        payload = estimate.payload_bytes or step.payload_bytes or 0
        link = (step.network, step.direction)
        self.inflight_bytes[link] += payload
        outstanding = self.inflight_bytes[link]
        self.peak_inflight_bytes = max(self.peak_inflight_bytes, outstanding)
        identity = (
            layer,
            step.direction,
            step.target_endpoint,
            tuple(sorted(set(block.context_lengths))),
        )
        tail = packet_tail(cfg, identity, payload, block.request_ids)
        # A bounded synthetic switch drops excess bursts and retries after a
        # configured round-trip. Queue service is accounted by the native wire.
        rounds = (
            max(0, math.ceil(outstanding / cfg.switch_buffer_bytes) - 1)
            if cfg.switch_buffer_bytes
            else 0
        )
        retry = rounds * cfg.retry_us / 1e6
        try:
            _, start, wire_end, priced = yield from wire()
            if tail + retry:
                yield self.env.timeout(tail + retry)
            arrival = self.env.now
        finally:
            self.inflight_bytes[link] -= payload
        if step.direction == "a_to_f":
            yield from self.receiver.arrived(key)
            blockers = yield from stream.arrived(flow)
        end = self.env.now
        row = dict(
            iteration=block.iteration_id,
            block=block.block_id,
            layer=layer,
            owner=block.placement.a_dp_rank,
            direction=step.direction,
            node_id=step.node_id,
            ready=ready,
            admitted=admitted,
            wire_start=start,
            wire_end=wire_end,
            arrival=arrival,
            released=end,
            payload_bytes=payload,
            tail_s=tail,
            retry_s=retry,
            retry_rounds=rounds,
            outstanding_bytes=outstanding,
            receive_wait_s=end - arrival,
            admission_wait_s=admitted - ready,
            receive_occupied=self.receiver.slots.count,
            receive_stream=stream_id,
            post_index=post_index,
            blocked_by=blockers,
            request_count=block.request_count,
        )
        self.records.append(row)
        details = dict(priced.details)
        details.update(
            synthetic_transport=row,
            resource_occupancy_end_at=details.get(
                "resource_occupancy_end_at", wire_end
            ),
            resource_occupancy_scope="wire_excludes_synthetic_flight_and_receive_wait",
        )
        return ready, start, end, replace(priced, details=details)
