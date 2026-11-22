# KV caching, transfer and storage

TokenSim models local prefix reuse, programmable cache control, prefill-to-decode KV transfer and external memory or flash tiers.

## Local prefix caching

With `--batching paged-attn`, requests reuse cached attention key/value (KV) blocks when they supply matching `hash_ids` for full input blocks.

```bash
tokensim \
  --batching paged-attn \
  --block_size 16 \
  --request_count 3 \
  --workload_type qwen_jsonl \
  --dataset_path ./dataset/prefix_reuse_example.jsonl \
  --cluster ./data/clusters/1_a100/h1.json \
  --model ./data/models/llama-7b.yaml \
  --qps 10 \
  --verbose none
```

A `qwen_jsonl` record supplies block hashes and optional namespace fields directly; a `json_pairs` record places them in its third element `[input_length, output_length, metadata]`.

```json
{
  "request_id": 1,
  "input_length": 48,
  "output_length": 4,
  "hash_ids": ["a", "b", "x"],
  "cache_salt": "optional-version",
  "reuse_group": "tenant-a"
}
```

Supply at least `floor(input_length / block_size)` entries in `hash_ids` to cover all full prompt blocks.
Lookup matches `(model, cache_salt, reuse_group)` and ordered block hashes starting from block 0, stopping at the earliest mismatch.
Even when every prompt block hits, the worker computes the final prompt token to sample the initial output token.
Released blocks retain their cached hashes until evicted in least-recently-used (LRU) order after uncached free blocks are exhausted.
Uniform caches publish prompt blocks once prompt processing finishes, and `output_hash_ids` publishes complete generated decode blocks when the prompt is block-aligned.
Grouped KV caches store multiple page granularities (such as full, sliding-window and compressed-attention groups) and reuse the longest prefix resident across every required group.

Result JSON files report `reuse_hit_blocks`, `reuse_miss_blocks`, `reuse_hit_tokens`, `effective_prefill_tokens` and `prefix_cache_hit_rate`.

## Programmable cache policies

Select a cache-control policy with `--cache_policy`:

- `apc` or `noop`: standard automatic prefix caching.
- `retain`: priority- and deadline-based retention (`--cache_retention_priority`, `--cache_retention_seconds`).
- `quota`: session-level GPU slot reservation across active, loading and waiting requests (`--cache_quota_fraction`, `--cache_quota_max_wait_seconds`).
- `movement`: idle-gap demotion and pre-arrival restoration over an asynchronous store (`--cache_prefetch_lead_seconds`, `--cache_prefetch_budget_bytes`).
- `retain_movement`, `all`, `forced_movement`, `enhanced_coop`: composed retention, quota and movement policies.

`all` and `enhanced_coop` gate demotion on 80% physical GPU occupancy and a 0.5-second idle gap; `forced_movement` uses a 30-second window.
`enhanced_coop` estimates effective pause windows from observed admission and queue service, and supports `cache_one_shot=true` metadata to bypass external writes and reclaim unreferenced ephemeral child pages.
Set `--kv_cache_memory_bytes` to cap per-rank physical KV memory on ordinary workers, or use `--kv_cache_capacity_tokens_per_dp_rank` or `--runtime_memory_profile` (see [Workloads](workloads.md#observed-runtime-kv-capacity)).

## Select a KV connector

Pass a connector file via `--kv_transfer_config` or set `kv_transfer` on the cluster or worker group.
Explicit worker-group `kv_transfer` blocks take precedence over `--kv_transfer_config`, which overrides the cluster default.

| Connector | Role | Layout and media | Timing model |
| --- | --- | --- | --- |
| `NoopConnector` (default), `P2PConnector` | Prefill-to-decode handoff | Live request pages | Topology link or configured endpoint fabric (see [Worker layouts](serving.md)) |
| `MooncakeConnector` | Direct prefill-to-decode handoff | Live request blocks | Blocking protocol latency (`1.5x` fixed latency across nodes) |
| `MooncakeStoreConnector` | Shared remote or embedded prefix store | Uniform (DRAM + optional SSD/plugin) or grouped (DRAM) | Uniform charges step-blocking I/O; grouped queues overlap compute |
| `UniformAsyncStore` | Queued external store for uniform KV (PP1) | Uniform DRAM, SSD or storage plugin | Queued restoration, read priority and rate-limited background writes |
| `SimpleCPUOffloadConnector` | Per-rank host DRAM offload | Grouped pages in host DRAM | Queued per-worker host DMA |
| `TieredKVConnector` | L1 GPU + per-rank L2 DRAM + shared L3 store | Grouped pages across DRAM/SSD tiers | Queued DMA, shared network and L3 media service |
| `MultiConnector` | Composed connectors (`connectors` list) | Compatible uniform connectors | Longest-prefix load winner; summed load/save costs |

Top-level connector fields are `kv_connector`, `kv_role` (`kv_producer`, `kv_consumer` or `kv_both`, inferred from worker role when omitted), `kv_connector_extra_config`, `kv_rank` and `engine_id`.

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

## Mooncake store and transfer timing

Engines with the same `master_server_address` share one store instance in the simulation environment.
Store keys bind the model geometry, page byte layout, tensor/pipeline partition, `tenant_id`, `cache_prefix`, `group_id`, `pcp_rank`, `dcp_rank`, `cache_salt`, `reuse_group` and prefix hash sequence.

| Option in `kv_connector_extra_config` | Default | Behavior |
| --- | --- | --- |
| `cache_layout` | `uniform` | `uniform` or `grouped`, matched to the worker's KV layout |
| `mode` | `embedded` | `embedded` (host link) or `standalone-store` (host link plus network hop) |
| `protocol` | `rdma` | `tcp` (40 us, 12.5 GiB/s), `rdma` (8 us, 50 GiB/s), `nvlink` (2 us, 300 GiB/s) or `nvmeof` (80 us, 7 GiB/s) |
| `global_segment_size` / `_gb` | 8 GiB | Per-rank contribution in `embedded` mode; total pool size in `standalone-store` mode |
| `memory_capacity_blocks` | unset | Explicit block capacity for uniform DRAM stores |
| `local_buffer_size` / `_gb` | 1 GiB | Per-rank transfer staging reservation |
| `host_available_bytes` | unset | Optional cap on pool plus staging memory |
| `admission_policy`, `eviction_policy` | `always`, `lru` | Admission (`always` or `never`) and eviction (`lru` or `none`) |
| `save_policy` | `mooncake` | `mooncake` (missing prompt blocks), `once` (all prompt blocks) or `every_step` |
| `load_async`, `transfer_overlap` | `false`, `false` | Asynchronous transfer lifecycle and compute overlap |
| `num_nics`, `parallel_paths`, `transfer_utilization` | `1`, `1`, `1.0` | Path parallelism `max(num_nics, parallel_paths)` and bandwidth utilization in `(0, 1]` |
| `fixed_latency_us`, `bandwidth_bytes_per_s` | unset | Global protocol overrides (`protocol_latency_us` / `protocol_bandwidth_bytes_per_s` override per protocol) |
| `enable_lookup` | `true` | Toggle client lookup while keeping its embedded DRAM contribution |
| `offload` | unset | Pluggable offload tier `{backend, name, config}` |
| `enable_offload`, `offload_tier`, `ssd_*` | `false`, `ssd` | Built-in SSD capacity, bandwidth and latency fields |

Grouped `MooncakeStoreConnector` runs on single-node PP1 deployments with `replica_num: 1`, byte-sized DRAM (`enable_offload: false`), `eviction_policy: "lru"`, `load_async: true`, `transfer_overlap: true` and `rdma` or `tcp`.

## Queued uniform and tiered stores

`UniformAsyncStore` adds a transaction queue with optional `online_policy` (`reuse_threshold`, `read_priority`, `write_bytes_per_second`, `save_batch_bytes`, `max_backlog_bytes` and `prefetch`) for uniform PP1 serving.
Set `aggregate_tensor_parallel: true` when one scheduler represents an entire TP replica's shard bytes.
Runs export `storage_io.jsonl`, `storage_request_io.jsonl`, `cache_control.jsonl` and `cache_control_summary.json`.

`TieredKVConnector` coordinates L1 GPU slots, per-rank L2 host DRAM (`l2_bytes_per_rank`, `staging_bytes_per_rank`) and a shared L3 service (`l3_bytes`, `backend`, `backend_bytes_per_second`, `backend_latency_us`, `network_bytes_per_second`, `network_latency_us`).
Supported `write_policy` modes are `write_through`, `write_back` (`high_watermark`, `low_watermark`, `watermark_eviction`), `write_around` and `selective_write_back`.

Result JSON files report `connector_*` transfer totals, `mooncake_*` store counters and `storage_metrics[master][instance]`.

## Storage backend plugins

Uniform Mooncake and `UniformAsyncStore` accept custom offload media via `offload` inside `kv_connector_extra_config`:

```json
{
  "offload": {
    "backend": "my_storage_package:ExampleBackend",
    "name": "flash",
    "config": {
      "capacity_bytes": 1099511627776,
      "bandwidth_bytes_per_s": 107374182400
    }
  }
}
```

`backend` accepts `module:Class` syntax or an alias registered via `StorageBackendFactory.register_backend(name, builder)`.
A backend implements `capacity_bytes`, `estimate_io(request: IORequest) -> IOCost` and `describe() -> dict` using contracts from `TokenSim.storage`:

```python
from TokenSim.config.validation import integer, number
from TokenSim.storage import IOCost, IORequest


class ExampleBackend:
    def __init__(self, config):
        if set(config) - {"capacity_bytes", "bandwidth_bytes_per_s"}:
            raise ValueError("unknown storage option")
        self.capacity_bytes = integer(config["capacity_bytes"], "capacity_bytes", minimum=1)
        self.bandwidth = number(config["bandwidth_bytes_per_s"], "bandwidth_bytes_per_s", positive=True)

    def estimate_io(self, request: IORequest) -> IOCost:
        return IOCost(request.nbytes / self.bandwidth)

    def describe(self) -> dict:
        return {
            "capacity_bytes": self.capacity_bytes,
            "bandwidth_bytes_per_s": self.bandwidth,
        }
```

See the [HBF plugin package](../../examples/storage/hbf/README.md) for a complete size-dependent bandwidth implementation.
