"""Fluid max-min sharing across independent send, receive and fabric budgets."""

from dataclasses import dataclass
import math

from TokenSim.errors import ConfigurationError


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ConfigurationError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive finite number")
    return float(value)


def fair_rates(paths, capacities):
    """Progressive filling freezes only flows using a saturated resource."""
    remaining = dict(capacities)
    rates = {key: 0.0 for key in paths}
    active = set(paths)
    while active:
        users = {
            resource: sum(resource in paths[key] for key in active)
            for resource in capacities
        }
        increment = min(
            remaining[resource] / count for resource, count in users.items() if count
        )
        for key in active:
            rates[key] += increment
        for resource, count in users.items():
            remaining[resource] = max(0.0, remaining[resource] - increment * count)
        saturated = {
            resource
            for resource, count in users.items()
            if count and remaining[resource] <= capacities[resource] * 1e-12
        }
        active = {key for key in active if not saturated.intersection(paths[key])}
    return rates


@dataclass
class Flow:
    source: int
    target: int
    bytes: int
    remaining: float
    started: float
    done: object


class EndpointFabric:
    def __init__(
        self,
        env,
        *,
        send_bytes_per_second,
        receive_bytes_per_second,
        fabric_bytes_per_second,
        launch_seconds=0.0,
    ):
        self.env = env
        self.send = positive(send_bytes_per_second, "send_bytes_per_second")
        self.receive = positive(receive_bytes_per_second, "receive_bytes_per_second")
        self.fabric = positive(fabric_bytes_per_second, "fabric_bytes_per_second")
        if not math.isfinite(launch_seconds) or launch_seconds < 0:
            raise ConfigurationError("launch_seconds must be finite and nonnegative")
        self.launch = launch_seconds
        self.active = {}
        self.rates = {}
        self.last_update = env.now
        self.changed = env.event()
        self.sequence = 0
        self.intervals = []
        self.completed = []
        env.process(self._run())

    def _advance(self):
        duration = self.env.now - self.last_update
        if duration and self.active:
            self.intervals.append(
                dict(start=self.last_update, end=self.env.now, rates=dict(self.rates))
            )
        for key, flow in self.active.items():
            flow.remaining = max(0.0, flow.remaining - duration * self.rates[key])
        self.last_update = self.env.now

    def _rebalance(self):
        paths = {
            key: (f"s{flow.source}", f"r{flow.target}", "fabric")
            for key, flow in self.active.items()
        }
        capacities = {
            resource: self.send if resource.startswith("s") else self.receive
            for path in paths.values()
            for resource in path
            if resource != "fabric"
        }
        capacities["fabric"] = self.fabric
        self.rates = fair_rates(paths, capacities) if paths else {}

    def _notify(self):
        event, self.changed = self.changed, self.env.event()
        event.succeed()

    def transfer(self, source, target, bytes_):
        if type(bytes_) is not int or bytes_ < 0:
            raise ConfigurationError("Transfer bytes must be a nonnegative integer")
        if self.launch:
            yield self.env.timeout(self.launch)
        self._advance()
        key = self.sequence
        self.sequence += 1
        flow = Flow(
            source, target, bytes_, float(bytes_), self.env.now, self.env.event()
        )
        self.active[key] = flow
        self._rebalance()
        self._notify()
        yield flow.done

    def _run(self):
        while True:
            if not self.active:
                yield self.changed
                continue
            delay = min(
                flow.remaining / self.rates[key] for key, flow in self.active.items()
            )
            # Round an unrepresentable positive tail to the next clock ULP.
            if delay > 0 and self.env.now + delay == self.env.now:
                delay = math.nextafter(self.env.now, math.inf) - self.env.now
            yield self.env.any_of([self.changed, self.env.timeout(max(0.0, delay))])
            self._advance()
            finished = [
                key
                for key, flow in self.active.items()
                if flow.remaining <= max(1e-5, flow.bytes * 1e-12)
            ]
            for key in finished:
                flow = self.active.pop(key)
                self.completed.append(
                    dict(
                        flow_id=key,
                        source=flow.source,
                        target=flow.target,
                        bytes=flow.bytes,
                        start=flow.started,
                        end=self.env.now,
                    )
                )
                flow.done.succeed()
            self._rebalance()

    def report(self):
        return dict(
            send_bytes_per_second=self.send,
            receive_bytes_per_second=self.receive,
            fabric_bytes_per_second=self.fabric,
            launch_seconds=self.launch,
            completed=self.completed,
            intervals=self.intervals,
            active_flows=len(self.active),
        )
