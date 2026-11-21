"""Minimal P/D layout and known service durations for transfer tests."""

from TokenSim.config.config import ClusterConfig, KVTransferConfig, WorkerGroupConfig
from TokenSim.config.workload_config import WorkloadConfig
from TokenSim.hardware.links import LinkCatalog
from TokenSim.latency.base import LatencyBackend
from TokenSim.results import MetricData
from tests.support.hardware import make_device, make_links, make_model

PREFILL_LATENCY = 0.0012

DECODE_LATENCY = 0.0023

_MODEL = make_model(
    "TestModel",
    hidden_size=1,
    intermediate_size=2,
    num_layers=1,
    num_attention_heads=1,
    vocab_size=4,
)

_DEVICE = make_device("TestGPU", memory_gib=0.0002)


class _DeterministicBackend(LatencyBackend):
    """Fixed service times so timing assertions stay exact."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def estimate_step_latency(self, requests):
        if requests[0].is_prefill or getattr(requests[0], "needs_recompute", False):
            self.calls.append(("prefill", len(requests)))
            return PREFILL_LATENCY
        self.calls.append(("decode", len(requests)))
        return DECODE_LATENCY


def _workload_config() -> WorkloadConfig:
    return WorkloadConfig(
        distribution="burst",
        prefill_mean_len=32,
        prefill_range_len=0,
        decode_mean_len=2,
        decode_range_len=0,
        decode_len_distribution="uniform",
        first_token_latency=MetricData(0, 0, 0),
        decode_token_latency=MetricData(0, 0, 0),
        qps=1,
    )


def _pd_cluster() -> ClusterConfig:
    return ClusterConfig(
        num_workers=2,
        networks={"net1": "ethernet-test"},
        worker_groups=[
            WorkerGroupConfig(
                role="prefill", hardware="TestGPU", num_workers=1, network="net1"
            ),
            WorkerGroupConfig(
                role="decode", hardware="TestGPU", num_workers=1, network="net1"
            ),
        ],
        kv_transfer=KVTransferConfig(kv_connector="P2PConnector"),
    )


def _expected_transfer_latency(bytes_: int) -> float:
    # Both workers share network net1, so the synthesized topology puts them in
    # one node connected by the device's scale-up link.
    link = LinkCatalog(make_links()).get("nvlink-test")
    return link.transfer_us(bytes_) * 1e-6
