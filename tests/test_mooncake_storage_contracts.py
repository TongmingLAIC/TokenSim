"""Mooncake capacity, transfer accounting, and shared-store isolation."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.config.config import KVTransferConfig, ParallelConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.kv_transfer.connectors.factory import KVConnectorFactory
from TokenSim.kv_transfer.connectors.mooncake_store import MooncakeStoreConnector
from TokenSim.kv_transfer.connectors.store_pages import (
    PageStore,
    _StoreView,
    _SharedService,
)
from TokenSim.kv_transfer.native_storage import (
    NativePageStore,
    validate_external_stores,
)
from TokenSim.llm.llm_request import Request
from TokenSim.mooncake.config import MooncakeConfig, parse_mooncake_config
from TokenSim.mooncake.pool_key import PoolKey, KeyMetadata
from TokenSim.mooncake.service import get_mooncake_service, shared_store


class MooncakeCapacityOnlyTest(unittest.TestCase):
    def test_disabled_lookup_is_client_local_and_preserves_shared_capacity(self):
        env = simpy.Environment()
        options = MooncakeConfig(global_segment_size=1000, local_buffer_size=10)
        owner = object()
        first = shared_store(env, options, ("grouped",), lambda: owner)
        disabled = replace(options, enable_lookup=False)
        self.assertIs(shared_store(env, disabled, ("grouped",), object), first)
        with self.assertRaises(ConfigurationError):
            MooncakeConfig(enable_lookup="false")

    def test_uniform_consumer_contributes_capacity_without_loading_or_saving(self):
        cache = SimpleNamespace(
            block_size=16, size_per_token=1, model="test", rank_info=None
        )
        config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={"enable_lookup": False},
        )
        connector = KVConnectorFactory.create_connector(config, cache)
        request = Request(1, 64, 2, 16, hash_ids=[1, 2, 3, 4])
        connector.service.store.put(connector._keys_for_request(request))
        self.assertEqual(connector.get_num_new_matched_tokens(request, 0), 0)
        request.scheduled_tokens = 16
        output = SimpleNamespace(running=[request], preempted=[])
        self.assertEqual(connector.build_connector_meta(output).saves, [])
        env = simpy.Environment()
        p = get_mooncake_service(
            config, block_bytes=16, owner=env, contributor=object()
        )
        d = get_mooncake_service(
            config, block_bytes=16, owner=env, contributor=object()
        )
        self.assertIs(p, d)
        self.assertEqual(d.store.memory_capacity_bytes, 2 * 8 * (1 << 30))

    def test_grouped_capacity_only_probe_does_not_touch_the_store(self):
        connector = PageStore.__new__(PageStore)
        connector.config = SimpleNamespace(kv_role="kv_consumer")
        connector.store_config = SimpleNamespace(enable_lookup=False)
        # No store/manager is installed: a capacity-only client must not use them.
        self.assertEqual(
            connector.probe(object(), SimpleNamespace(hit_tokens=32)), (32, ())
        )


class UniformReplicaStorageTest(unittest.TestCase):
    def test_full_replica_objects_charge_capacity_and_transfer_bytes(self):
        cache = SimpleNamespace(
            block_size=16,
            size_per_token=64,
            model="fixture",
            rank_info=None,
            parallel_config=ParallelConfig(tensor_parallel_size=4),
        )
        options = dict(
            mode="standalone-store",
            global_segment_size=16384,
            aggregate_tensor_parallel=True,
            master_server_address="replica-accounting-test",
        )
        config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector", kv_connector_extra_config=options
        )
        connector = MooncakeStoreConnector(config, cache)
        self.assertEqual(connector.block_bytes, 4096)
        self.assertEqual(connector.service.store.block_bytes, 4096)
        keys = [PoolKey(KeyMetadata("fixture"), str(i)) for i in range(2)]
        connector.service.store.put(keys)
        self.assertEqual(connector.service.store.memory_used_bytes, 8192)
        plan = connector._build_load_plan(
            Request(0, 32, 2, 16), dict(blocks=2, keys=keys, tier="memory")
        )
        self.assertEqual(plan.bytes, 8192)

    def test_unsupported_layout_and_nonboolean_flag_fail_at_boundary(self):
        for options in [
            dict(cache_layout="grouped", aggregate_tensor_parallel=True),
            dict(aggregate_tensor_parallel=1),
        ]:
            with self.assertRaises(ConfigurationError):
                parse_mooncake_config(options)


class CountedStore(NativePageStore):
    def __init__(self):
        super().__init__(1024)
        self.validations = 0

    def validate(self):
        self.validations += 1
        return super().validate()


class SharedStoreValidationTest(unittest.TestCase):
    def test_shared_views_and_independent_rank_stores_all_receive_validation(self):
        shared, left, right = CountedStore(), CountedStore(), CountedStore()
        views = [_StoreView(shared, (rank,)) for rank in range(24)]
        clients = [SimpleNamespace(store=s) for s in [*views, left, right]]
        self.assertEqual(validate_external_stores(clients), 3)
        self.assertEqual([s.validations for s in [shared, left, right]], [1, 1, 1])

    def test_shared_corruption_still_fails_the_drain(self):
        shared = CountedStore()
        shared.used_bytes = 1
        clients = [
            SimpleNamespace(store=_StoreView(shared, (rank,))) for rank in range(8)
        ]
        with self.assertRaises(SimulationStateError):
            validate_external_stores(clients)
        self.assertEqual(shared.validations, 1)


class SharedNamespaceTest(unittest.TestCase):
    def service(self):
        worker = SimpleNamespace(
            engine=SimpleNamespace(workers=[None]), env=simpy.Environment()
        )
        return _SharedService(worker, MooncakeConfig())

    def test_shared_views_reuse_ready_pages_and_keep_model_scopes_isolated(self):
        service = self.service()
        left = _StoreView(service.store, service.namespace_token(("model-a",)))
        right = _StoreView(service.store, service.namespace_token(tuple(["model-a"])))
        other = _StoreView(service.store, service.namespace_token(("model-b",)))
        page = left.reserve_write((0, "prefix", 1), 64)
        self.assertIsNone(right.lookup((0, "prefix", 1)))
        left.finish_write(page)
        self.assertIs(right.lookup((0, "prefix", 1)).page, page.page)
        self.assertIsNone(other.lookup((0, "prefix", 1)))
        self.assertEqual(service.store.used_bytes, 64)
