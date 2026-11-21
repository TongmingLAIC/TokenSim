"""Independent DP busy waves over native complete-model execution domains."""

from TokenSim.errors import ConfigurationError
from TokenSim.llm.dp_coordinator import DPStepCoordinator


class DomainCoordinators:
    def __init__(self, coordinators):
        self.coordinators = coordinators
        self.by_worker = {
            worker: coordinator
            for coordinator in coordinators.values()
            for worker in coordinator.worker_ids
        }
        self.worker_ids = set(self.by_worker)

    def consume_messages(self, worker):
        yield from self.by_worker[worker.id].consume_messages(worker)

    def notify(self):
        for coordinator in self.coordinators.values():
            coordinator.notify()

    def reset_stats(self):
        for coordinator in self.coordinators.values():
            coordinator.reset_stats()

    def describe(self):
        return dict(
            mode="independent_serving_domains",
            domains={
                name: coordinator.describe()
                for name, coordinator in self.coordinators.items()
            },
        )


def build_dp_coordinator(engine):
    groups = {}
    for worker in engine.request_workers:
        if engine._is_schedulable_worker(worker):
            groups.setdefault(worker.execution_domain, []).append(worker)
    coordinators = {}
    for domain, workers in groups.items():
        config = workers[0].parallel_config
        if config.data_parallel_size > 1:
            if config.pipeline_parallel_size != 1:
                raise ConfigurationError(
                    "DP lockstep pipeline stages are not yet supported"
                )
            coordinators[domain] = DPStepCoordinator(
                engine.env,
                engine,
                workers,
                sync_interval=engine.config.dp_sync_interval,
            )
    if not coordinators:
        return None
    if set(coordinators) == {None}:
        return coordinators[None]
    return DomainCoordinators(coordinators)
