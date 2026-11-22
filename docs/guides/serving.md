# Parallelism and worker layouts

TokenSim configures cluster serving through `worker_groups`, `parallel_config` and hierarchical interconnect topologies.

## Worker roles

| Role | Scheduling behavior |
| --- | --- |
| `hybrid` | Executes both prompt processing (prefill) and token generation (decode) |
| `prefill` | Executes prefill and hands off KV state to decode workers |
| `decode` | Receives KV state and executes decode |

Select pool placement policies with `--prefill_worker_pool_type` (default `round_robin`) and `--decode_worker_pool_type` (default `least_gpu_memory`; `balanced_load` is also supported).

## Tensor, pipeline, data and expert parallelism

Configure shared-layout parallelism in the cluster JSON or override individual axes on the CLI:

```bash
tokensim \
  --batching paged-attn \
  --request_count 8 \
  --cluster ./data/clusters/8_a100/tp2_pp2_dp2.json \
  --model ./data/models/llama-7b.yaml \
  --qps 10 \
  --verbose none
```

When multiple ranks are configured in a shared layout, `sum(worker_groups[*].num_workers)` equals `tensor_parallel_size * pipeline_parallel_size * data_parallel_size`.
Global rank order varies tensor parallelism (TP) fastest, followed by pipeline parallelism (PP), then data parallelism (DP).

| Field in `parallel_config` | Default | Meaning |
| --- | --- | --- |
| `tensor_parallel_size` | `1` | Intra-layer TP shard count per replica |
| `pipeline_parallel_size` | `1` | Contiguous layer stages per replica |
| `data_parallel_size` | `1` | Replica count |
| `data_parallel_rank`, `data_parallel_size_local` | `0`, `1` | Recorded deployment rank and DP ranks per node |
| `tensor_parallel_collective_latency_scale` | `1.0` | Multiplier on TP collective latency |
| `pipeline_parallel_activation_latency_scale` | `1.0` | Multiplier on PP stage-transfer latency |
| `enable_expert_parallel` | `false` | Distribute mixture-of-experts (MoE) routed experts across ranks |
| `expert_parallel_scope` | `global` | `global` (`TP * DP` stage ranks) or `per_dp` (each replica's `TP` ranks) |
| `expert_parallel_size` | scope size | Ranks per expert-parallel (EP) group |
| `expert_placement_strategy` | `linear` | `linear` contiguous blocks or `round_robin` |
| `all2all_backend` | `allgather_reducescatter` | `allgather_reducescatter`, `naive`, `deepep_high_throughput` or `deepep_low_latency` |

- **Tensor parallelism**: shards projections, attention heads, dense FFNs and KV capacity across TP ranks, adding post-attention and post-FFN all-reduces (replaced by EP dispatch/combine on EP MoE layers).
- **Pipeline parallelism**: partitions layers across PP stages (assigning remainder layers to early stages), charges point-to-point activation transfers between stages and places the vocabulary output head on the final stage.
- **Data parallelism**: schedules requests on each replica's `TP=0, PP=0` leader within the smallest rank KV pool.
  Use `--sticky_sessions`, `--router_policy consistent_hash` or `--router_policy fixed_mapping --router_mapping_path FILE` for conversation affinity.
  `--dp_lockstep` synchronizes steps across hybrid PP1 replicas (checking completion every `--dp_sync_interval` steps, default 16).

## Independent prefill and decode domains

Dedicated `prefill` and `decode` worker groups can each declare their own `parallel_config` and `kv_transfer` when top-level `parallel_config` is omitted:

```json
{
  "num_workers": 16,
  "topology": "gb200_nvl72",
  "networks": {"fabric": "ethernet800Gb"},
  "worker_groups": [
    {
      "name": "P",
      "role": "prefill",
      "hardware": "gb200",
      "num_workers": 8,
      "network": "fabric",
      "parallel_config": {
        "data_parallel_size": 8,
        "data_parallel_size_local": 8,
        "enable_expert_parallel": true,
        "expert_parallel_scope": "global",
        "expert_parallel_size": 8
      }
    },
    {
      "name": "D",
      "role": "decode",
      "hardware": "gb200",
      "num_workers": 8,
      "network": "fabric",
      "parallel_config": {
        "data_parallel_size": 8,
        "data_parallel_size_local": 8,
        "enable_expert_parallel": true,
        "expert_parallel_scope": "global",
        "expert_parallel_size": 8
      }
    }
  ],
  "kv_transfer": {
    "kv_connector": "NoopConnector",
    "kv_connector_extra_config": {
      "pd_endpoint_limits": {
        "send_bytes_per_second": 50000000000,
        "receive_bytes_per_second": 50000000000,
        "fabric_bytes_per_second": 400000000000
      }
    }
  }
}
```

Collectives stay isolated within each domain, and P/D handoff routes completed prefills into the destination domain's DP space.
With `pd_endpoint_limits`, `EndpointFabric` shares sender, receiver and fabric bandwidth across active transfers while pinning source pages and reserving receiver slots until DMA completes.
Prefill samples the initial output token on completion; handoff transfer and decode scheduling contribute to the subsequent inter-token interval.
Results report per-domain layouts and transport flows under `parallel_serving_domains`, `moe_serving_domains` and `pd_transport`.

## Topologies and communication reporting

Clusters reference hierarchical topologies in `data/topologies/*.yaml` (with optional `device_indices`) or synthesize a two-level node/network hierarchy from `networks` and `data/topologies/links.yaml`.
Results record `parallel_config`, `parallel_groups`, `parallel_link_type_counts`, `parallel_comm_match_type_counts` and per-phase synchronization latencies.
See [Operator latency and data packages](../reference/operator-latency.md#communication-models) for collective lookup and analytical formulas.
