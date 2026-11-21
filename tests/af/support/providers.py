"""Small physical-topology providers for cache and wire plan checks."""

from TokenSim.af.providers.operator_table import TopologyAFTransferProvider
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.hardware.placement import build_topology_placement
from tests.af.support.config import _cluster_dict
from tests.af.support.execution import af_test_hardware


def sample_transfer_provider():
    cluster = ClusterConfig(**_cluster_dict(attention_workers=2, ffn_workers=2))
    hardware = af_test_hardware()
    workers = cluster.workers()
    placement = build_topology_placement(cluster, workers, hardware)
    return TopologyAFTransferProvider(hardware, cluster, workers, placement)
