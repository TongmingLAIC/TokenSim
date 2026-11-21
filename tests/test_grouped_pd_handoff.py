"""Grouped P/D copies preserve actual pages, causal metadata and DMA ownership."""

import unittest

from TokenSim.llm.llm_request import Request


from tests.support.grouped_handoff import setup


class GroupedPDHandoffTest(unittest.TestCase):
    def test_receiver_reuses_proven_local_pages_and_copies_only_the_missing_tail(self):
        env, engine, handoff, req, failures = setup()
        source, target = engine.workers
        destination = target.scheduler.block_manager
        previous = Request(2, 33, 4, 16, hash_ids=[0, 1])
        destination.allocate(previous)
        for count in [16, 16, 1]:
            destination.ensure_slots(previous, count)
            previous.num_computed_tokens += count
            destination.cache_computed(previous)
        destination.free(previous)
        cold_pages = len(source.scheduler.block_manager.block_table.get_blocks(req.id))
        handoff.dispatch(source, [req])
        source.scheduler.running.clear()
        env.run()
        self.assertEqual(failures, [])
        transfer = next(
            event for event in handoff.events if event["kind"] == "transfer_start"
        )
        self.assertGreater(transfer["receiver_reused_pages"], 0)
        self.assertLess(transfer["receiver_allocated_pages"], cold_pages)
        self.assertEqual(
            transfer["receiver_reused_pages"] + transfer["receiver_allocated_pages"],
            cold_pages,
        )
        self.assertEqual(
            transfer["bytes"],
            transfer["receiver_allocated_pages"]
            * destination.layout.physical_slot_bytes,
        )
        destination.validate_conservation()

    def test_receiver_capacity_blocks_dma_until_physical_pages_are_released(self):
        env, engine, handoff, req, failures = setup(slots=12)
        source, target = engine.workers
        destination = target.scheduler.block_manager
        occupied = Request(2, 80, 4, 16, hash_ids=list(range(5)))
        destination.allocate(occupied)
        for _ in range(5):
            destination.ensure_slots(occupied, 16)
            occupied.num_computed_tokens += 16
        handoff.dispatch(source, [req])
        source.scheduler.running.clear()
        env.run(until=env.now + 0.01)
        self.assertFalse(handoff.fabric.completed)
        self.assertNotIn(req.id, destination.tables)
        self.assertTrue(source.scheduler.block_manager.transfer_references)
        destination.release_request_blocks(occupied)
        handoff.notify()
        env.run()
        self.assertEqual(failures, [])
        self.assertTrue(handoff.fabric.completed)
        self.assertEqual(target.scheduler.loaded_waiting, [req])
        destination.validate_conservation()

    def test_actual_transfer_keeps_unpublished_tails_private_and_resumes_without_recompute(
        self,
    ):
        env, engine, handoff, req, failures = setup(target_inflight=16)
        source, target = engine.workers
        original = req.admission_state()
        source_manager, destination = (
            source.scheduler.block_manager,
            target.scheduler.block_manager,
        )
        proven = set(source_manager.cache)
        count = sum(
            index * spec.block_tokens < req.num_computed_tokens
            for spec in source_manager.groups
            for index in source_manager.tables[req.id][spec.group_id]
        )
        handoff.dispatch(source, [req])
        source.scheduler.running.clear()
        self.assertTrue(source_manager.transfer_references)
        source_manager.validate_conservation()
        env.run()
        self.assertEqual(failures, [])
        self.assertEqual(
            sum(flow["bytes"] for flow in handoff.fabric.completed),
            count * destination.layout.physical_slot_bytes,
        )
        self.assertEqual(set(destination.cache), proven)
        self.assertEqual(req.admission_state()[1:], original[1:])
        self.assertFalse(source_manager.transfer_references)
        self.assertFalse(destination.transfer_references)
        self.assertTrue(
            any(not b.cached for b in destination.block_table.get_blocks(req.id))
        )
        chunks = req.prefill_chunks
        batch, _ = target.scheduler.schedule()
        self.assertEqual(batch, [req])
        self.assertFalse(req.is_context_build)
        self.assertEqual(req.num_computed_tokens, 33)
        self.assertEqual(handoff.jobs, {})
        env.run(until=env.now + 0.01)
        req.advance(env, 0.01, 1)
        self.assertEqual(req.prefill_chunks, chunks)
        destination.validate_conservation()

    def test_failed_dma_releases_private_destinations_before_reporting_failure(self):
        env, engine, handoff, req, failures = setup()
        source, target = engine.workers

        def failed(*args):
            yield env.timeout(0.01)
            raise ValueError("injected DMA failure")

        handoff.fabric.transfer = failed
        handoff.dispatch(source, [req])
        source.scheduler.running.clear()
        env.run()
        self.assertEqual([str(error) for error in failures], ["injected DMA failure"])
        self.assertEqual(handoff.jobs, {})
        self.assertEqual(handoff.events[-1]["kind"], "transfer_failed")
        self.assertFalse(req.owner_worker_ids)
        self.assertFalse(target.scheduler.loaded_waiting)
        self.assertFalse(target.scheduler.block_manager.cache)
        for worker in engine.workers:
            manager = worker.scheduler.block_manager
            self.assertFalse(manager.transfer_references)
            self.assertEqual(manager.get_num_allocated_blocks(), 1)
            manager.validate_conservation()

    def test_cancellation_waits_for_dma_and_publishes_no_target_prefix(self):
        env, engine, handoff, req, failures = setup()
        source, target = engine.workers
        handoff.dispatch(source, [req])
        source.scheduler.running.clear()
        env.run(until=env.now + 0.01)
        req.request_cancel(env)
        handoff.cancel(req)
        self.assertTrue(source.scheduler.block_manager.transfer_references)
        self.assertTrue(target.scheduler.block_manager.transfer_references)
        env.run()
        self.assertEqual(failures, [])
        self.assertEqual(target.scheduler.block_manager.cache, {})
        self.assertEqual(handoff.jobs, {})
        for worker in engine.workers:
            manager = worker.scheduler.block_manager
            self.assertFalse(manager.transfer_references)
            self.assertEqual(manager.get_num_allocated_blocks(), 1)
            manager.validate_conservation()


if __name__ == "__main__":
    unittest.main()
