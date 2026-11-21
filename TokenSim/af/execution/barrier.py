"""Controlled layer-entry peer barrier with unchanged routing and return format."""

from collections import Counter
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.errors import ConfigurationError, SimulationStateError


class PeerLayerBarrier:
    def __init__(self, executor):
        self.executor = executor
        self.env = executor.env
        self.enabled = executor.cluster_config.af.peer_layer_barrier
        self.peers = executor.cluster_config.af_domains[0].parallel.data_parallel_size
        self.failure = self.env.event()
        self.sequences = Counter()
        self.iterations = {}
        self.rounds = {}

    def check(self):
        if self.enabled and self.failure.triggered:
            raise self.failure.value

    def register(self, iteration):
        if not self.enabled:
            return
        self.check()
        owner = iteration.blocks[0].placement.a_dp_rank
        epoch = self.sequences[owner]
        phases = tuple(b.forward_phase for b in iteration.blocks)
        state = self.rounds.setdefault(
            epoch, dict(phases=phases, gates={}, finished=set())
        )
        if phases != state["phases"]:
            error = ConfigurationError(
                "peer layer barrier requires matching microbatch counts and phases"
            )
            self.abort(error)
            raise error
        self.iterations[iteration.iteration_id] = (epoch, owner)
        self.sequences[owner] += 1

    def wait(self, block, layer, previous):
        if not self.enabled:
            return previous
        self.check()
        epoch, owner = self.iterations[block.iteration_id]
        gates = self.rounds[epoch]["gates"]
        gate = gates.setdefault(
            (block.block_id, layer), dict(arrivals={}, event=self.env.event())
        )
        if owner in gate["arrivals"]:
            raise SimulationStateError("duplicate peer layer barrier arrival")
        ready = self.env.now
        gate["arrivals"][owner] = tuple(previous)
        if len(gate["arrivals"]) == self.peers:
            gate["event"].succeed()
        yield self.env.any_of((gate["event"], self.failure))
        self.check()
        node = f"i{block.iteration_id}:b{block.block_id}:l{layer}:peer_layer_barrier"
        if self.executor.trace.enabled:
            self.executor.trace.record(
                AFTraceEvent(
                    iteration_id=block.iteration_id,
                    block_id=block.block_id,
                    layer_id=layer,
                    phase=block.phase,
                    event_type="synchronize",
                    operation="peer_layer_barrier",
                    resource_id=f"peer_barrier:{epoch}:{block.block_id}:{layer}",
                    node_id=node,
                    predecessor_ids=tuple(previous),
                    ready_at=ready,
                    start_at=self.env.now,
                    end_at=self.env.now,
                    queue_duration=self.env.now - ready,
                    service_duration=0,
                    request_count=block.request_count,
                    token_count=block.token_count,
                    source_id="runtime:peer-layer-barrier",
                    dataset_version="runtime",
                    match_type="analytical",
                    details={
                        "peers": self.peers,
                        "round": epoch,
                        "peer_predecessor_ids": tuple(
                            p for ids in gate["arrivals"].values() for p in ids
                        ),
                    },
                    source_domain=block.placement.a_domain,
                    dp_rank=owner,
                )
            )
        return (node,) if self.executor.trace.enabled else previous

    def complete(self, iteration):
        identity = self.iterations.pop(iteration.iteration_id, None)
        if identity is None or self.failure.triggered:
            return
        epoch, owner = identity
        state = self.rounds[epoch]
        state["finished"].add(owner)
        if len(state["finished"]) == self.peers:
            del self.rounds[epoch]

    def abort(self, reason):
        if self.enabled and not self.failure.triggered:
            self.failure.succeed(
                SimulationStateError(f"AF peer layer barrier aborted: {reason}")
            )
            self.rounds.clear()
            self.iterations.clear()
