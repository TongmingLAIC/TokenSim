# KV transfer and storage

Connectors control how simulated workers move attention key/value (KV) cache pages between prefill and decode workers or reuse them through an external store.
Choose a connector by cache geometry, media and timing requirements.

## Select a connector

| Connector | Use | Layout and media | Timing |
| --- | --- | --- | --- |
| `NoopConnector` (default), `P2PConnector` | Prefill-to-decode handoff | Live request pages; topology timing for P2P | See [worker layouts](serving.md) |
| `MooncakeConnector` | Direct prefill-to-decode handoff | Live request blocks | Blocking transfer estimate |
| `MooncakeStoreConnector` | Prefix-addressed remote or embedded cache | Uniform: host memory with optional SSD or plugin; grouped: host memory | Uniform blocks on I/O; grouped queues overlap compute |
| `UniformAsyncStore` | Queued external cache for ordinary serving, PP1 | Uniform, with SSD or a storage plugin | Asynchronous restoration, background writes |
| `SimpleCPUOffloadConnector` | Local host offload per rank | Grouped pages in DRAM | Queued host DMA |
| `TieredKVConnector` | Local host cache plus one shared remote storage service | Grouped L2 DRAM and L3 DRAM or SSD model | Queued DMA, network and L3 service |
| `MultiConnector` | Compose compatible connectors | For example, direct transfer plus uniform Mooncake | Per-child transfer policies |

Pass a connector JSON file with `--kv_transfer_config`, as in the following example.

```bash
tokensim \
  --batching paged-attn \
  --request_count 8 \
  --cluster ./data/clusters/8_a100/p2d5.json \
  --kv_transfer_config ./data/kv_transfer/mooncake_p2p.json \
  --model ./data/models/llama-7b.yaml \
  --qps 10 \
  --verbose none
```

You can also put the same configuration under `kv_transfer` in a cluster JSON file.
A command-line file replaces the cluster-level connector default.
A worker group with an explicit `kv_transfer` keeps its own configuration.
The following table describes the top-level fields.

| Field | Default | Meaning |
| --- | --- | --- |
| `kv_connector` | `NoopConnector` | Connector class name |
| `kv_role` | inferred | `kv_producer`, `kv_consumer` or `kv_both`; when omitted, `prefill` workers become `kv_producer`, `decode` workers `kv_consumer` and `hybrid` workers `kv_both` |
| `kv_connector_extra_config` | `{}` | Connector options (configuration reference below) |
| `kv_rank` | worker id | Rank recorded in transfer plans; each worker uses its own id |
| `engine_id` | random UUID | Engine identity, shared by the engine's workers |

Unknown top-level fields and unknown Mooncake connector options are rejected.

## Direct Prefill-to-Decode Transfer

Use `MooncakeConnector` with a cluster that assigns prefill and decode to separate worker groups.
The connector calculates the bytes in the transferred KV blocks and estimates the GPU-to-GPU transfer time using the rules in [Transfer Timing](#transfer-timing).
The direct Mooncake formula applies its cross-node fixed-latency multiplier when source and target workers differ.

The included `data/kv_transfer/mooncake_p2p.json` selects RDMA.
This connector does not require `hash_ids` because it transfers the live request's blocks between the selected workers.
Without a connector, the cluster transfers blocks over the link at the smallest topology level containing both workers, as described in [Parallelism](serving.md).

## Shared Store and Prefix Reuse

Use `MooncakeStoreConnector` with a workload that supplies `hash_ids` to identify reusable prompt blocks.

```bash
tokensim \
  --batching paged-attn \
  --block_size 16 \
  --request_count 4 \
  --cluster ./data/clusters/1_a100/h1.json \
  --kv_transfer_config ./data/kv_transfer/mooncake_store_standalone_ssd.json \
  --dataset_path ./dataset/mooncake_reuse.json \
  --workload_type json_pairs \
  --model ./data/models/llama-7b.yaml \
  --qps 50 \
  --verbose none
```

An embedded store uses memory on the compute node, and a standalone store is modeled as a separate store reached over the network.
The standalone example keeps one block in memory and up to 64 on SSD.
`data/kv_transfer/mooncake_store_embedded.json` configures an embedded store with 128 memory blocks and no SSD tier.
In this single-worker cluster, the worker's GPU cache serves repeated prefixes that are still in GPU memory.
The store serves blocks that have left GPU memory or requests running on another worker, as shown in the MultiConnector example below.

Store keys identify the model, the byte layout of cached data and its tensor-parallel (TP) and pipeline-parallel (PP) partitions.
They also include `tenant_id`, `cache_prefix`, `group_id`, `pcp_rank`, `dcp_rank`, `cache_salt`, `reuse_group` and hashes that encode each block's preceding prefix.
A lookup returns only the longest contiguous prefix.
The worker checks its GPU cache before the store and loads only the additional blocks it needs.
Even when the whole prompt is cached, the worker computes at least the last prompt token to produce an output token.
With `save_policy: mooncake`, the last prompt block is excluded from store loads.

`save_policy` controls which computed blocks are saved.

- `mooncake`, the default, saves full prompt blocks missing from the store after prompt processing finishes.
  It skips saving in a step that loaded blocks from the store and in all decode steps.
- `once` follows the same timing as `mooncake` and transfers all full prompt blocks without checking whether they are already stored.
- `every_step` saves computed full prompt blocks on every scheduled step, including decode steps.

Grouped storage (`cache_layout: grouped`) supports only `mooncake`.
The block manager makes newly computed pages available to the store when they are retained in the cache.

### Store Sharing

Stores belong to the SimPy simulation environment.
Engines using the same `master_server_address` share a store, even if their `engine_id` or data-parallel (DP) rank differs.
Each embedded engine contributes its ranks' memory segments once.
The store sums these contributions and the clients' transfer buffers using each client's configured byte limits.
A standalone store has one shared capacity limit and an additional transfer buffer for each client.
Clients can use different write admission rules, save policies, asynchronous settings, cache namespaces and buffer sizes within the same store.
Transfer settings can also differ, except for standalone grouped stores, where those settings define one shared network queue.
Clients sharing a store must agree on its mode, cache layout, eviction policy, `memory_capacity_blocks`, standalone capacity, replica count, offload media settings and host memory limit.
Separate environments start with fresh stores.

A cache hit requires matching model dimensions, data layout, model partition, tenant, prefix and block hashes.
Uniform stores require one byte layout for all pages and reject incompatible layouts or settings.
A uniform cache object becomes available for reuse after its transfer and any required offload-media write finish.
Reads and writes protect the objects they are using from eviction until the modeled transfer completes.
Generated tokens with no known content identifiers and partial prompts use internal request identifiers.
Matching public request IDs from different engines therefore cannot make those blocks reusable.

See [Prefix Cache](prefix-cache.md) for workload formats and exact hash matching rules.

## Cache Layouts and Capacity

`cache_layout: uniform`, the default, stores one object per KV block and uses the same page layout for every block.
Set `cache_layout: grouped` for models with multiple page types, such as sliding-window and compressed attention caches.
TokenSim checks this setting against the engine's actual KV layout.
`data/kv_transfer/mooncake_store_dram_agentx.json` is a grouped example whose memory budget is explained in the [AgentX guide](workloads.md).
Grouped mode of `MooncakeStoreConnector` requires all of the following settings.

- Use one compute node and one pipeline stage (PP1).
- Keep one replica in CPU memory with `replica_num: 1` and `enable_offload: false`, and specify capacity in bytes without `memory_capacity_blocks`.
- Set `eviction_policy: lru`, `load_async: true` and `transfer_overlap: true`.
- Use `protocol: rdma` or `protocol: tcp` and leave `preferred_segment` unset.
- Select the connector directly, outside a `MultiConnector`.

In embedded mode, `global_segment_size` is each rank's contribution to the shared KV store.
In standalone mode, it is the separate store's total capacity.
A uniform store uses `memory_capacity_blocks` instead when it is set.
Each rank also reserves `local_buffer_size` bytes for transfers, in addition to the KV store's capacity.
When `host_available_bytes` is supplied, the store and all clients' transfer buffers must fit within that limit.
Integer size fields count bytes, and the `global_segment_size_gb` and `local_buffer_size_gb` aliases count gibibytes (GiB), where 1 GiB = 2^30 bytes.

## MooncakeStoreConnector memory and SSD tiers

Uniform Mooncake supports optional storage backend plugins through the generic `offload` object.
See [storage backend plugins](../development/storage-backends.md) for the interface and configuration.
The [optional high-bandwidth flash (HBF) package](../../examples/storage/hbf/) shares the store's admission, eviction, readiness and prefix-matching rules with SSD.
Its media model and independent GPU memory examples are documented in [HBF example](../../examples/storage/hbf/README.md).

With `eviction_policy: "lru"`, memory pressure evicts the least recently used object that no transfer is using.
With the SSD tier enabled, every accepted object is also written to SSD, and that persistence write must finish before the operation completes.
An object evicted from memory stays readable from SSD, with any missing SSD copy written during eviction.
With the SSD tier disabled, eviction removes the object from the store.
With `eviction_policy: "none"`, a write that does not fit in memory is rejected, or goes directly to SSD when the tier is enabled.
`enable_offload: false` disables the SSD tier and keeps CPU memory storage and prefix reuse enabled.
`enable_lookup: false` skips store lookup for that client while retaining its embedded DRAM contribution.
A `kv_consumer` with lookup disabled contributes capacity without loading or publishing KV pages.

For uniform storage, enable the SSD tier with the following configuration.

```json
{
  "enable_offload": true,
  "offload_tier": "ssd",
  "ssd_capacity_blocks": 64,
  "ssd_read_bandwidth_bytes_per_s": 7516192768,
  "ssd_write_bandwidth_bytes_per_s": 3221225472,
  "ssd_read_latency_us": 100,
  "ssd_write_latency_us": 200
}
```

`ssd_capacity_blocks` takes precedence over `ssd_capacity_bytes`, and an enabled SSD tier requires positive capacity.
An SSD read takes its fixed latency plus the transferred bytes divided by read bandwidth.
Writes use the same formula with write bandwidth and must finish before persistence completes.
SSD eviction removes the least recently used cache object, identified by its block key.

## Transfer Timing

| Path | Timing |
| --- | --- |
| `MooncakeConnector` | `fixed_latency + bytes / (bandwidth * transfer_utilization * paths)`, where `paths = max(num_nics, parallel_paths)`; cross-node transfers multiply the fixed latency by 1.5 |
| Uniform store, embedded | The device's host link (`interconnect.host_link`, typically PCIe): link latency plus `bytes / (bandwidth * transfer_utilization)` |
| Uniform store, standalone | The host-link time plus a network hop at the protocol timing, its bandwidth capped by the worker's network link |
| Grouped store, embedded | Queued per-worker host-link transfers (separate read and write queues) |
| Grouped store, standalone | Serial host-link and shared network queues; the network queue is capped by the node's network link and shared by all workers |

`TokenSim/mooncake/transfer_engine.py` defines the default protocol latencies and bandwidths listed below.
Override all protocols with `fixed_latency_us` and `bandwidth_bytes_per_s`, or individual protocols with `protocol_latency_us` and `protocol_bandwidth_bytes_per_s`.

| Protocol | Fixed latency | Bandwidth |
| --- | --- | --- |
| `tcp` | 40 us | 12.5 GiB/s |
| `rdma` | 8 us | 50 GiB/s |
| `nvlink` | 2 us | 300 GiB/s |
| `nvmeof` | 80 us | 7 GiB/s |

Grouped storage models contention on its queues and overlaps transfers with compute.
The direct connector and the uniform mode of `MooncakeStoreConnector` charge blocking transfer latencies per step.
`UniformAsyncStore` uses a separate asynchronous transaction queue.
SSD timing is modeled as an independent storage tier alongside the network protocol.
Transfer buffers count toward host memory capacity.
The included uniform and SSD configuration files use compact block limits to demonstrate multi-tier caching and eviction behavior.

## Mooncake configuration reference

These options configure the Mooncake connectors.
`UniformAsyncStore` also uses the uniform store options with the overlap and topology requirements described below.

| Field | Default | Supported values or behavior |
| --- | --- | --- |
| `cache_layout` | `uniform` | `uniform` or `grouped`, matched against the engine's KV layout |
| `master_server_address` | `default` | Identity of the simulated shared store; no real connection is made |
| `tenant_id`, `cache_prefix` | `default`, `""` | Logical cache namespaces, independent of engine/DP identity |
| `mode` | `embedded` | `embedded` or `standalone-store` |
| `protocol` | `rdma` | `tcp`, `rdma`, `nvlink` or `nvmeof` |
| `global_segment_size` / `_gb` | 8 GiB | Memory-store capacity (see [Cache Layouts and Capacity](#cache-layouts-and-capacity)) |
| `local_buffer_size` / `_gb` | 1 GiB | Per-rank staging memory, additional to the KV pool |
| `memory_capacity_blocks` | unset | Uniform store: explicit capacity in blocks |
| `host_available_bytes` | unset | Upper bound for pool plus staging |
| `admission_policy` | `always` | `always` admits, `never` rejects writes |
| `eviction_policy` | `lru` | `lru` or `none` |
| `save_policy` | `mooncake` | `mooncake`, `once` or `every_step` |
| `replica_num` | 1 | Storage models one replica; other counts are rejected |
| `load_async` | `false` | Marks load and save plans asynchronous; a store releases a request's blocks after its asynchronous save completes. Direct transfers and uniform Mooncake still charge the transfer time to the step |
| `transfer_overlap` | `false` | Requires `load_async`; grouped Mooncake and `UniformAsyncStore` require it; uniform Mooncake and `MooncakeConnector` reject it |
| `num_nics`, `parallel_paths` | 1, 1 | Effective path count is their maximum |
| `fixed_latency_us`, `bandwidth_bytes_per_s` | unset | Protocol timing overrides for every protocol |
| `protocol_latency_us`, `protocol_bandwidth_bytes_per_s` | `{}` | Per-protocol timing overrides, keyed by protocol name |
| `transfer_utilization` | 1.0 | Fraction of peak transfer bandwidth, in `(0, 1]`; the bundled grouped example uses 0.75 |
| `preferred_segment` | unset | Segment label stored with admitted objects (uniform only) |
| `group_id`, `pcp_rank`, `dcp_rank` | 0 | Additional pool-key namespaces |
| `offload` | unset | Optional backend reference, instance name and plugin-owned configuration |
| `enable_lookup` | `true` | Enable prefix lookup for this client; independent of its DRAM contribution |
| `enable_offload`, `offload_tier` | `false`, `ssd` | Legacy SSD offload switch and tier |
| `ssd_capacity_blocks`, `ssd_capacity_bytes` | unset, 0 | SSD capacity; blocks take precedence |
| `ssd_read_bandwidth_bytes_per_s`, `ssd_write_bandwidth_bytes_per_s` | 7 GiB/s, 3 GiB/s | SSD bandwidth |
| `ssd_read_latency_us`, `ssd_write_latency_us` | 100, 200 | SSD fixed latency |

Capacity fields ending in `_bytes` are byte counts; bandwidth fields ending in `_bytes_per_s` are bytes per second.
Conflicting size aliases, fractional byte budgets and invalid rates are rejected.
The cluster and serving configuration determine parallelism.

## MultiConnector

`data/kv_transfer/mooncake_multi_connector.json` combines direct transfer and an embedded store on a cluster with separate prefill and decode workers.

```bash
tokensim \
  --batching paged-attn \
  --block_size 16 \
  --request_count 4 \
  --cluster ./data/clusters/8_a100/p2d5.json \
  --kv_transfer_config ./data/kv_transfer/mooncake_multi_connector.json \
  --dataset_path ./dataset/mooncake_reuse.json \
  --workload_type json_pairs \
  --model ./data/models/llama-7b.yaml \
  --qps 50 \
  --verbose none
```

Each connector listed in `connectors` contributes its own transfer plans and metrics.
The `connectors` list is the only option on `MultiConnector` itself.
Prefix matching uses the longest prefix reported by those connectors, and transfer timing adds their load and save costs.
Only the connector reporting that longest prefix loads the request and updates its cached token count.
Each transfer plan returns to the connector that created it, including when connectors have the same type or are nested.
Child connectors inherit the parent's worker role, rank and engine identifier unless explicitly overridden.
Grouped storage is unsupported inside `MultiConnector`.

## Queued uniform storage

`UniformAsyncStore` requires uniform KV and pipeline parallel size one.
It defaults to `load_async: true` and `transfer_overlap: true`.
Use `aggregate_tensor_parallel: true` when a single ordinary tensor-parallel scheduler represents the replica's physical shards; the store and I/O queue then use aggregate object bytes.
Clients sharing one master must use the same online queue policy, and they require a master separate from blocking uniform Mooncake clients.

The following connector file selects a standalone SSD-backed store with foreground read priority and a bounded write backlog.
Configure the optional [HBF plugin](../../examples/storage/hbf/README.md) in `offload` in place of the legacy SSD fields to queue HBF media costs.

```json
{
  "kv_connector": "UniformAsyncStore",
  "kv_connector_extra_config": {
    "mode": "standalone-store",
    "memory_capacity_blocks": 0,
    "enable_offload": true,
    "ssd_capacity_bytes": 1099511627776,
    "online_policy": {
      "reuse_threshold": 2,
      "read_priority": true,
      "write_bytes_per_second": 16000000000,
      "save_batch_bytes": 268435456,
      "max_backlog_bytes": 8589934592
    }
  }
}
```

Set media capacity in bytes and bandwidths in bytes per second.
`backend_includes_transport` defaults to true; set it to false and supply `fixed_latency_us` and `bandwidth_bytes_per_s` when transport needs a separate service stage.
The lane services one transaction at a time.
Read priority reorders waiting transactions while active ones finish.
A save batch must fit the configured burst and backlog limits and hold at least one physical object.
Writes await credit before taking the lane; batches exceeding the backlog limit are dropped.

Foreground restoration reserves destinations and holds separate DMA references until I/O finishes.
Completion publishes only copied known pages.
Cancellation releases DMA references after any admitted I/O completes and suppresses publication.
Reuse admission uses past prefix arrivals and may also admit a computed parent prefix on a tool-wait signal.
Optional `online_policy.prefetch` estimates launch time from observed tool pauses, uses known stored pages, and applies both the configured byte budget and the normal GPU allocation gate.
Prefetch reuse requires the copied pages and content epochs to remain valid at later admission.

`storage_io.jsonl` and `storage_request_io.jsonl` record queue, service and cancellation intervals.
`cache_control.jsonl` and `cache_control_summary.json` account for useful and wasted prefetch bytes.
Load exposure subtracts overlap with completed local forward intervals.
`prefill_wait_s` is time to first token minus counted prefill service; it includes admission, scheduling and restoration.
`local_forward_busy_seconds` is the union of forward intervals on the representative scheduler.

## Grouped host offload and tiered storage

`SimpleCPUOffloadConnector` gives each rank a private CPU memory pool for grouped pages.
Set `cpu_bytes_to_use_per_rank` and optionally `host_available_bytes` and `transfer_utilization` inside `kv_connector_extra_config`.

`TieredKVConnector` uses the GPU allocator as L1, private host DRAM as L2, and a shared L3 storage service for compatible clients in the same simulation environment.
Related clients can share L3 content with matching model, cache-group, TP/PP and shard scopes.
Configure the per-rank budget with `l2_bytes_per_rank`, including `staging_bytes_per_rank` (64 MiB by default).
Set `l3_bytes`, `backend: "dram"` or `"ssd"`, `backend_bytes_per_second`, `backend_latency_us`, and the shared `network_bytes_per_second` and `network_latency_us` in `kv_connector_extra_config`.
GPU DMA and L2 use padded slot bytes; L3 capacity uses grouped payload bytes.
The model represents one shared network budget; it does not provide separate per-node staging or network budgets.

| `write_policy` | Behavior |
| --- | --- |
| `write_through` | Populate L2 and enqueue L3 persistence, pinning the local source until completion. |
| `write_back` | Copy missing pages before eviction or at the high watermark, then reclaim eligible L2 pages toward the low watermark. |
| `write_around` | Write to L3 through bounded host staging. |
| `selective_write_back` | Use write-back while excluding pages marked by explicit one-shot application hints. |

`high_watermark` and `low_watermark` default to 0.8 and 0.6.
Pending eviction bytes count toward the target, and a new read protects a pending victim.
Set `watermark_eviction: false` to copy eligible pages without reclaiming them at the watermark.
`promote_reads` controls admission on L3 reads.
Allocation enforces physical capacity while copies are pending.
A ready external copy becomes reusable after its required transfers complete.

The CLI drains accepted I/O and reports a separate persistence audit when tail drain is enabled.
The audit may evict L3 pages when the store is full; policy traffic and audit traffic are counted separately.
Reported fixed-window throughput and service-level-objective goodput exclude the audit.
Profiling counters subtract the warmup baseline when available; lifetime peak gauges include warmup and drain.
Coverage reports distinguish per-rank L2 bytes, distinct ready objects and payload overlap across tiers.

## Metrics

Fields beginning with `connector_` report transfer counts, blocks, bytes and time spent waiting.
Fields beginning with `mooncake_` report store reads and writes, hits and misses, memory and offload hit counts, legacy SSD operations, transfer and store latency, and effective bandwidth.
`storage_metrics` reports media operations for every backend, grouped by shared master and storage instance.
They also record accepted and rejected writes, evictions, pending asynchronous jobs, hit tokens from GPU, memory and disk, sampled cache keys and the effective offload configuration.

Operation counters belong to each connector, even when its store is shared.
Exporting or resetting one engine's profile does not include or clear another engine's operations.
Resetting counters preserves cached data and the references protecting objects used by pending transfers.
