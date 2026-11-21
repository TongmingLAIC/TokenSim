# Queued uniform storage

`UniformAsyncStore` reuses the native Mooncake object store, prefix keys and byte-based SSD/HBF backends with asynchronous GPU restoration and background writes.
It supports uniform KV, PP1 and ordinary decode.
Set `aggregate_tensor_parallel` when one ordinary TP scheduler represents all physical KV shards of its replica.
The connector charges their combined object bytes to the store and I/O queue while its allocation gate checks the representative rank's corresponding cache geometry.

```json
{
  "kv_connector": "UniformAsyncStore",
  "kv_connector_extra_config": {
    "mode": "standalone-store",
    "memory_capacity_blocks": 0,
    "aggregate_tensor_parallel": true,
    "enable_offload": true,
    "ssd_capacity_bytes": 24000000000000,
    "ssd_read_bandwidth_bytes_per_s": 109400000000,
    "ssd_write_bandwidth_bytes_per_s": 46885714285.71429,
    "ssd_read_latency_us": 100,
    "ssd_write_latency_us": 200,
    "online_policy": {
      "reuse_threshold": 2,
      "read_priority": true,
      "write_bytes_per_second": 16000000000,
      "write_burst_bytes": 8589934592,
      "max_backlog_bytes": 34359738368,
      "save_batch_bytes": 268435456,
      "prefetch": true,
      "prefetch_budget_bytes": 17179869184,
      "prefetch_lead_seconds": 0.1,
      "history_entries": 1000000,
      "history_seconds": 300,
      "backend_includes_transport": true
    }
  }
}
```

The numerical values above define an analytical policy example.
The selected media's effective transfer path must match the bandwidth and latency supplied to its backend.
With `backend_includes_transport`, the backend cost includes that path and the connector adds no separate transport delay.
Otherwise provide explicit `fixed_latency_us` and `bandwidth_bytes_per_s` for an additional transfer stage.
Replace the legacy SSD fields with `offload` and the existing HBF plugin to select HBF media.
Queued clients sharing one master must agree on their online queue policy and cannot share that master with metadata-only uniform clients.

## Admission and physical ownership

A uniform request reserves its complete context before any external load starts.
Both request references and DMA references protect its GPU destinations during I/O.
Successful completion registers only copied, known prefix pages and returns the request to the native scheduler for a real compute query.
The aligned Mooncake policy leaves the trailing prompt block for computation.
A cancelled load completes its admitted I/O, releases its DMA references and publishes no copied GPU pages.
Writes become available according to the existing store's modeled completion timestamps.

The shared transaction lane serializes submitted I/O batches.
Read priority reorders transactions waiting for the lane; active transactions complete without preemption.
`save_batch_bytes` sets the maximum compound write transaction and must fit both the write burst and backlog limits.
An object's physical size must fit the save batch.
Write credit is acquired before a background transaction takes the I/O lane, so foreground reads can use the lane while writes await credit.
The write budget charges KV writeback bytes as a conservative upper bound on NAND programming bytes.
The connector drops a background batch when accepting it would exceed the byte backlog bound.

## Observable policy state

Reuse admission uses a bounded, age-limited history of prefixes observed in arriving requests.
A current parent tool-wait signal can admit that parent's already computed prefix even before its repetition threshold is reached.
This signal gives a policy useful state without exposing the eventual tool completion time.
The controller saves only pages whose current content identities still match the observed snapshot.

Prefetch uses the most recently completed parent prefix and an exponential average of previous observed pause durations.
Its launch estimate subtracts the known media service cost and configured lead interval.
It waits for already queued writes of that known prefix when the prefix is not yet visible in the store.
The controller does not inspect a future request, future hash identifiers or future trace timestamps.
Each copy remains subject to a byte budget and the normal GPU allocation gate.
Control requests allocate their known prefix plus one private query page and never enter user request execution or throughput counts.

The controller records actual copied pages with their physical content epochs.
It credits a prefetch only when a later genuine admission uses the same pages and epochs.
An eviction and later reload of the same key cannot receive that credit.
Completed or cancelled I/O without later page reuse remains visible in the prefetch waste counters.

## Metrics and checks

`storage_io.jsonl` records transaction queue, credit, service and completion times with request and worker identifiers.
`storage_request_io.jsonl` records admitted request-load intervals and cancellation outcomes.
`cache_control.jsonl` records control-copy actions, and `cache_control_summary.json` reports copied/useful/wasted bytes, media reuse and per-worker load exposure.
Exposure removes overlap with that worker's completed forward intervals; its per-request sum is separate from a system makespan.
The request export also retains prefill/decode service aggregates and residual time outside that counted service.
`prefill_wait_s` equals TTFT minus prefill service for requests that emitted a token and is null for cancelled partial prefills.
The residual includes admission, scheduling and asynchronous restoration and does not identify storage waiting alone.
Per-request service includes the forward intervals in which that request participates, so concurrent requests can count the same batch interval.
`local_forward_busy_seconds` reports the union of completed local forward intervals once per scheduler, including the representative scheduler of an ordinary TP replica.
These fields observe existing timing state without changing scheduling or simulated latency.
The native backend's `storage_metrics` retain operation counts, bytes and service work.
The normal CLI drain validates storage capacity and unfinished I/O ownership.
The tests also verify that GPU DMA references survive cancellation and return to zero after completion.

```bash
python -m unittest tests.test_storage_queue tests.test_uniform_async_store
```
