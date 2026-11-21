"""Cache movement, transfer ownership, and cooperative retention policies."""

from types import SimpleNamespace as NS
import unittest

import simpy

from TokenSim.block.hybrid.kv_layout import CacheLayerPage, CacheGroup, HybridKVLayout
from TokenSim.block.hybrid.manager import HybridKVBlockManager
from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.kv_transfer.connectors.simple_cpu import AsyncOffloadConnector
from TokenSim.kv_transfer.storage.native import NativePageStore, QueuedKVLink
from TokenSim.kv_transfer.policies.movement import MovementRuntime, PrefixSnapshot
from TokenSim.kv_transfer.policies.child import reclaim_child, skip_write
from TokenSim.llm.llm_request import Request


class CacheMovementTest(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        page = CacheLayerPage("layer.0", "full", 16, 8)
        layout = HybridKVLayout((CacheGroup(0, (page,)),), 16, 512, 32, 0, 32, 128)
        self.manager = HybridKVBlockManager(layout, 16, "toy")
        config = KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both"
        )
        self.connector = AsyncOffloadConnector(config, NS(hybrid_layout=layout))
        scheduler = NS(
            block_manager=self.manager,
            running=[],
            waiting=[],
            loaded_waiting=[],
            loading={},
        )
        self.connector.env = self.env
        self.connector.scheduler = scheduler
        self.connector.manager = self.manager
        self.connector.worker = NS(engine=NS(dp_coordinator=None))
        self.connector.changed = self.env.event()
        self.connector.store = NativePageStore(512 * 32, allocation_unit=512)
        self.connector.read_link = QueuedKVLink(self.env, 512.0)
        self.connector.write_link = QueuedKVLink(self.env, 1e9)
        self.manager.publish_callbacks.append(self.connector._queue_write)
        self.previous = self.request(0)
        self.manager.allocate(self.previous)
        self.manager.ensure_slots(self.previous, 64)
        self.previous.num_computed_tokens = 64
        self.manager.cache_computed(self.previous)
        self.snapshot = PrefixSnapshot.capture(
            self.previous, self.manager, self.connector
        )
        self.manager.free(self.previous)
        self.connector.wait_for_save()
        self.env.run(until=0.001)
        self.events = []
        controller = NS(
            env=self.env,
            engine=NS(
                config=NS(
                    cache_prefetch_budget_bytes=4096, cache_prefetch_lead_seconds=0.1
                )
            ),
            owner=lambda req: req.chat_id,
            record=lambda kind, **values: self.events.append((kind, values)),
            prefetch_completed=lambda scheduler, req: None,
        )
        self.movement = MovementRuntime(controller)
        self.assertGreater(self.movement.move_ready(self.snapshot), 0)

    def request(self, ident):
        return Request(
            ident,
            64,
            1,
            16,
            hash_ids=[1, 2, 3, 4],
            chat_id="parent",
            cache_salt="scope",
            reuse_group="tenant",
        )

    def test_failed_write_releases_gpu_and_discards_unpublished_store_page(self):
        def fail(_):
            yield self.env.timeout(0)
            raise RuntimeError("copy failed")

        occupied = self.connector.store.used_bytes
        req = Request(2, 16, 1, 16, hash_ids=[99])
        self.manager.allocate(req)
        self.manager.ensure_slots(req, 16)
        req.num_computed_tokens = 16
        self.manager.cache_computed(req)
        self.manager.release_request_blocks(req)
        self.connector.write_link.transfer = fail
        self.connector.wait_for_save()
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            self.env.run()

        self.assertEqual(self.connector.write_jobs, 0)
        self.assertEqual(self.connector.store.used_bytes, occupied)
        self.assertFalse(self.manager.transfer_references)
        self.manager.validate_conservation()
        self.connector.store.validate()

    def start(self):
        state = dict(request=None, phase="prefetch", done=self.env.event())
        process = self.env.process(self.movement.copy(self.snapshot, state))
        return process, state

    def test_copy_becomes_visible_after_real_queued_completion(self):
        process, state = self.start()
        self.env.run(until=1.0)
        self.assertFalse(process.triggered)
        self.assertTrue(self.manager.transfer_references)
        self.assertEqual(self.manager.plan_reuse(self.request(100)).hit_tokens, 0)
        self.env.run(until=process)
        self.assertEqual(self.manager.plan_reuse(self.request(101)).hit_tokens, 48)
        self.assertFalse(self.manager.transfer_references)
        self.assertEqual(self.manager.get_num_allocated_blocks(), 1)
        self.assertFalse(self.connector.scheduler.loaded_waiting)

    def test_cancelled_copy_keeps_dma_pins_and_never_publishes(self):
        process, state = self.start()
        self.env.run(until=1.0)
        state["request"].cancel_requested = True
        self.assertTrue(self.manager.transfer_references)
        self.env.run(until=process)
        self.assertEqual(self.manager.plan_reuse(self.request(102)).hit_tokens, 0)
        self.assertFalse(self.manager.transfer_references)
        self.assertEqual(self.manager.get_num_allocated_blocks(), 1)
        self.assertEqual(self.movement.stats["prefetch_cancelled"], 1)

    def test_byte_budget_rejects_before_target_allocation(self):
        self.movement.controller.engine.config.cache_prefetch_budget_bytes = 1
        process, state = self.start()
        self.env.run(until=process)
        self.assertEqual(self.connector.read_link.jobs, 0)
        self.assertEqual(self.manager.get_num_allocated_blocks(), 1)
        self.assertFalse(self.manager.requests)
        self.assertEqual(self.movement.stats["prefetch_budget_rejected"], 1)

    def test_live_request_reference_prevents_move_reclamation(self):
        consumer = self.request(5)
        self.manager.allocate(consumer)
        self.manager.ensure_slots(consumer, 64)
        consumer.num_computed_tokens = 64
        self.manager.cache_computed(consumer)
        self.connector.wait_for_save()
        self.env.run(until=0.002)
        snapshot = PrefixSnapshot.capture(consumer, self.manager, self.connector)
        before = self.manager.get_num_allocated_blocks()
        self.assertEqual(self.movement.move_ready(snapshot), 0)
        self.assertEqual(self.manager.get_num_allocated_blocks(), before)
        self.manager.free(consumer)

    def test_shutdown_cancels_future_timers_without_allocating_pages(self):
        self.movement.snapshots["parent"] = self.snapshot
        self.movement.estimates["parent"] = 10.0
        self.movement.tool_wait(self.previous)
        self.env.run(until=2.0)
        self.assertTrue(self.movement.active)
        self.movement.close()
        self.env.run(until=2.01)
        self.assertFalse(self.movement.active)
        self.assertEqual(self.connector.read_link.jobs, 0)
        self.assertEqual(self.manager.get_num_allocated_blocks(), 1)


class CooperationTest(unittest.TestCase):
    def runtime(self, mode):
        env = simpy.Environment()
        controller = NS(
            mode=mode,
            env=env,
            owner=lambda req: req.chat_id,
            engine=NS(config=NS(cache_prefetch_lead_seconds=0.1)),
            record=lambda *a, **k: None,
        )
        return MovementRuntime(controller)

    def wait(self, mode, estimate, occupied):
        runtime = self.runtime(mode)
        calls = []
        runtime.move_ready = lambda snap: calls.append("move")

        def copy(snap, state):
            calls.append("copy")
            yield runtime.env.timeout(0)

        runtime.copy = copy
        snapshot = NS(
            manager=NS(occupied_fraction=lambda: occupied / 100),
            connector=NS(notify=lambda: None),
        )
        state = dict(
            snapshot=snapshot, phase="moving", done=runtime.env.event(), job_id=1
        )
        runtime.env.process(runtime._wait(state, estimate))
        runtime.env.run()
        return calls

    def test_movement_gates_on_wait_and_pressure_unless_forced(self):
        for mode, estimate, occupied, moves in (
            ("all", None, 99, False),
            ("all", 0.25, 99, False),
            ("all", 1, 79, False),
            ("enhanced_coop", 1, 79, False),
            ("all", 1, 80, True),
            ("forced_movement", 30, 10, True),
        ):
            with self.subTest(mode=mode, estimate=estimate, occupied=occupied):
                expected = ["move", "copy"] if moves else []
                self.assertEqual(self.wait(mode, estimate, occupied), expected)

    def test_tool_and_actual_admission_observations_are_separate(self):
        runtime = self.runtime("enhanced_coop")
        runtime.waiting["p"] = 0
        req = NS(chat_id="p")
        runtime.env.run(until=2)
        runtime.resuming(req)
        self.assertEqual(runtime.estimates["p"], 2)
        self.assertIn("p", runtime.waiting)
        runtime.env.run(until=5)
        runtime.resuming(req, admitted=True)
        self.assertEqual(runtime.admitted_estimates["p"], 5)
        self.assertNotIn("p", runtime.waiting)

    def test_child_reclamation_preserves_parent_and_live_blocks(self):
        page = CacheLayerPage("layer.0", "full", 16, 8)
        layout = HybridKVLayout((CacheGroup(0, (page,)),), 16, 512, 32, 0, 32, 128)
        manager = HybridKVBlockManager(layout, 8, "toy")
        requests = [Request(i, 16, 1, 16, hash_ids=[i]) for i in range(3)]
        blocks = []
        for req in requests:
            manager.allocate(req)
            manager.ensure_slots(req, 16)
            req.num_computed_tokens = 16
            manager.cache_computed(req)
            blocks.extend(manager.block_table.get_blocks(req.id))
        for req in requests[:2]:
            manager.release_request_blocks(req)
        keys = [block.block_hash for block in blocks]
        slots = [block.block_number for block in blocks]
        controller = NS(
            child_write_admission=True,
            stats={},
            record=lambda *a, **k: None,
            movement=NS(
                snapshots={"p": NS(manager=manager, blocks=((blocks[0], keys[0]),))}
            ),
        )
        req = NS(id=9, cache_one_shot=True)
        self.assertFalse(skip_write(controller, manager, blocks[0], req))
        self.assertTrue(skip_write(controller, manager, blocks[1], req))
        controller.child_reclamation = False
        reclaim_child(controller, NS(block_manager=manager), req, slots)
        self.assertEqual([block.block_hash for block in blocks], keys)
        controller.child_reclamation = True
        reclaim_child(controller, NS(block_manager=manager), req, slots)
        self.assertEqual(
            [block.block_hash for block in blocks], [keys[0], None, keys[2]]
        )
        manager.validate_conservation()
        req.cache_one_shot = False
        self.assertFalse(skip_write(controller, manager, blocks[1], req))

    def test_unannotated_trace_does_not_reveal_future_child_lifetime(self):
        import json
        from pathlib import Path
        from TokenSim.workload.agentx_support.parser import parse_trace_graph
        from TokenSim.errors import WorkloadValidationError

        path = Path(__file__).resolve().parents[1] / "fixtures/agentx_weka.jsonl"
        raw = json.loads(path.read_text().splitlines()[0])
        self.assertFalse(
            any(c.cache_one_shot for c in parse_trace_graph(raw).conversations)
        )
        child = next(r for r in raw["requests"] if r["type"] == "subagent")
        child["cache_one_shot"] = True
        with self.assertRaises(WorkloadValidationError):
            parse_trace_graph(raw)
        child["requests"] = child["requests"][:1]
        self.assertTrue(
            any(c.cache_one_shot for c in parse_trace_graph(raw).conversations)
        )

    def test_parent_content_cache_expires_when_a_session_releases(self):
        from TokenSim.kv_transfer.policies.child import parent_content

        runtime = self.runtime("enhanced_coop")
        manager = object()
        runtime.snapshots["p"] = NS(manager=manager, blocks=((None, "parent"),))
        controller = NS(movement=runtime)
        self.assertEqual(parent_content(controller, manager), {"parent"})
        runtime.release("p")
        self.assertEqual(parent_content(controller, manager), set())

    def test_component_switches_preserve_shared_content_checks(self):
        controller = NS(
            mode="enhanced_coop", child_write_admission=False, movement=None
        )
        req = NS(cache_one_shot=True)
        self.assertFalse(
            skip_write(controller, object(), NS(block_hash="private"), req)
        )
        runtime = self.runtime("enhanced_coop")
        runtime.controller.queue_window_enabled = False
        runtime.waiting["p"] = 0
        runtime.env.run(until=2)
        runtime.resuming(NS(chat_id="p"))
        self.assertNotIn("p", runtime.waiting)
        self.assertEqual(runtime.estimates["p"], 2)
