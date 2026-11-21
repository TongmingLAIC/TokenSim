"""Reads reorder queued writes; active I/O and write credit retain physical time."""

import unittest
import simpy

from TokenSim.kv_transfer.storage.online_config import OnlineStoragePolicy
from TokenSim.kv_transfer.storage.queue import StorageQueue


class StorageQueueTest(unittest.TestCase):
    def test_foreground_read_waits_for_active_write_and_passes_queued_write(self):
        env = simpy.Environment()
        queue = StorageQueue(env, OnlineStoragePolicy(read_priority=True))

        def write(delay, nbytes, duration):
            yield env.timeout(delay)
            self.assertTrue(queue.reserve_write(nbytes))
            yield env.process(queue.execute("write", nbytes, lambda: duration))

        def read():
            yield env.timeout(2)
            yield env.process(queue.execute("read", 1, lambda: 1))

        env.process(write(0, 10, 10))
        env.process(write(1, 10, 2))
        env.process(read())
        env.run()
        self.assertEqual(
            [(e["kind"], e["started_at"], e["completed_at"]) for e in queue.events],
            [("write", 0, 10), ("read", 10, 11), ("write", 11, 13)],
        )
        self.assertEqual(queue.backlog_bytes, 0)

    def test_credit_wait_leaves_media_available_for_reads(self):
        env = simpy.Environment()
        policy = OnlineStoragePolicy(
            read_priority=True,
            write_bytes_per_second=10,
            write_burst_bytes=10,
            max_backlog_bytes=20,
            save_batch_bytes=10,
        )
        queue = StorageQueue(env, policy)
        for _ in range(2):
            self.assertTrue(queue.reserve_write(10))
            env.process(queue.execute("write", 10, lambda: 0.1))
        self.assertFalse(queue.reserve_write(1))

        def read():
            yield env.timeout(0.2)
            yield env.process(queue.execute("read", 1, lambda: 0.1))

        env.process(read())
        env.run()
        self.assertEqual([e["kind"] for e in queue.events], ["write", "read", "write"])
        self.assertEqual(queue.events[-1]["started_at"], 1)
        self.assertEqual(queue.events[1]["started_at"], 0.2)
        self.assertEqual(queue.dropped_write_bytes, 1)


if __name__ == "__main__":
    unittest.main()
