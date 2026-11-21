"""Online lifecycle observations shared by uniform storage policy clients."""

from TokenSim.errors import ConfigurationError
from TokenSim.reporting.cache_control import CacheControlLog, write_uniform_summary
from TokenSim.kv_transfer.policies.uniform_prefetch import UniformPrefetch
from TokenSim.kv_transfer.io_observation import request_record


class UniformCacheController:
    def __init__(self, engine, connectors):
        self.engine = engine
        self.env = engine.env
        self.connectors = connectors
        self.mode = "uniform_online_storage"
        self._log = CacheControlLog(self.env, self.mode)
        self.changed = self.env.event()
        self.closed = False
        self.prefetch = UniformPrefetch(self)
        self.owners = {}
        self.stats = dict(completions=0, cancellations=0)
        self.request_rows = []
        for connector in connectors:
            connector.scheduler.cache_controller = self

    @staticmethod
    def owner(req):
        return str(req.chat_id or req.cache_salt or req.id)

    def notify(self):
        event, self.changed = self.changed, self.env.event()
        if not event.triggered:
            event.succeed()

    def completed(self, scheduler, req):
        self.stats["completions"] += 1
        self.request_rows.append(request_record(req, self.env.now, "successful"))
        self.owners.pop(req.id, None)
        if req.measurement_phase == "profile" and not req.cancel_requested:
            self.prefetch.capture(scheduler, req)

    def defer_admission(self, scheduler, req):
        self.owners[req.id] = scheduler
        return self.prefetch.blocks(scheduler, req)

    def resumed(self, req):
        scheduler = self.owners.get(req.id)
        if scheduler is not None:
            self.prefetch.resumed(scheduler, req)

    def client_resuming(self, req):
        self.prefetch.client_resuming(req)

    def tool_wait(self, req):
        self.prefetch.tool_wait(req)

    def play_finished(self, identity):
        self.prefetch.finish_play(identity)

    def cancelled(self, req):
        self.stats["cancellations"] += 1
        self.request_rows.append(request_record(req, self.env.now, "cancelled"))
        self.owners.pop(req.id, None)
        self.prefetch.release(self.owner(req))

    def close(self):
        self.closed = True
        self.prefetch.close()

    @property
    def pending(self):
        return self.prefetch.pending

    def configure_log(self, directory):
        self._log.configure_log(directory)

    def record(self, kind, **values):
        self._log.record(kind, **values)

    def write_summary(self, directory):
        write_uniform_summary(self, directory)


def attach_uniform_controller(engine):
    from TokenSim.kv_transfer.connectors.uniform_async import UniformAsyncStore

    connectors = [
        w.connector
        for w in engine.request_workers
        if isinstance(w.connector, UniformAsyncStore)
    ]
    if not connectors:
        return
    if engine.cache_controller is not None:
        raise ConfigurationError(
            "Uniform online storage uses its own lifecycle controller"
        )
    engine.cache_controller = UniformCacheController(engine, connectors)
