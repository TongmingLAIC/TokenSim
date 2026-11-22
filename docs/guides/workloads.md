# Workloads and AgentX replay

`tokensim` drives simulations from synthetic length distributions, request traces or AgentX multi-agent conversation trees.
Pass arrival rates, length distributions and service-level objectives (SLOs) on the CLI or through `--workload_config`, with CLI flags taking precedence.

| Workload type | Input format |
| --- | --- |
| `synthetic` | Prompt and output lengths sampled from configured distributions |
| `json_pairs` | JSON array of `[input_length, output_length]` pairs with optional metadata in element 3 |
| `qwen_jsonl` | JSON Lines records with `input_length`, `output_length` and optional timestamps or hashes |
| `agentx_weka` | AgentX JSON Lines traces of parent and subagent request trees |

For `json_pairs` and `qwen_jsonl`, set `--dataset_path`, `--request_count` and optional `--dataset_skip_count`.
Scale recorded arrival timestamps with `--trace_timestamp_scale` or target an arrival rate with `--trace_target_qps`.

```bash
tokensim \
  --model llama-7b \
  --batching paged-attn \
  --block_size 16 \
  --workload_type qwen_jsonl \
  --dataset_path dataset/qwen_example.jsonl \
  --request_count 3 \
  --qps 50 \
  --results_path results/qwen-example
```

See [KV caching, transfer and storage](kv-storage.md) for `hash_ids` prefix reuse and [README.md](../../README.md#workload) for latency targets.

## AgentX trace replay

Install the optional AgentX extras and download the pinned WEKA trace dataset:

```bash
python -m pip install -e '.[agentx]'
hf download semianalysisai/cc-traces-weka-062126 traces.jsonl \
  --repo-type dataset \
  --revision 23f152f6f0f9399a85901b89a6458def0ef16729 \
  --local-dir dataset/agentx
echo '29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276  dataset/agentx/traces.jsonl' | sha256sum --check
```

Run a replay against DeepSeek-V4-Pro with multi-token prediction (MTP):

```bash
tokensim \
  --workload_type agentx_weka \
  --dataset_path dataset/agentx/traces.jsonl \
  --model deepseek-v4-pro \
  --cluster data/clusters/8_b200/dep8.json \
  --batching paged-attn \
  --block_size 256 \
  --max_num_batched_tokens 8192 \
  --operator_backend vllm \
  --attention_backend module \
  --mhc_backend module \
  --moe_backend deep_gemm_amxf4_mega_moe \
  --mtp_num_speculative_tokens 3 \
  --mtp_acceptance_length 2.49 \
  --agentx_trace_count 1 \
  --agentx_concurrency 1 \
  --agentx_profile_duration 60 \
  --no-agentx_warmup \
  --results_path results/agentx/subset
```

In `closed_loop` mode (default), each turn issues once its parent/predecessor turns finish and its recorded inter-turn tool delay elapses, enforcing `SPAWN` branch activation and `JOIN` barriers while idle caps bound long quiet intervals.
`--agentx_issue_mode recorded` schedules turns at trace-relative timestamps after dependencies complete.
Use `--agentx_root_arrivals schedule.jsonl` (`{"time_s": 0.0, "trace_index": 0}` rows in nondecreasing order) with `--no-agentx_warmup` to launch agent trees on an open-loop schedule.

At the end of `--agentx_profile_duration`, replay stops admitting new work, drains active turns for `--agentx_grace_period` and cancels remaining turns within `--agentx_cancellation_timeout`.
`--sim_time` enforces a hard simulation-time cutoff (`termination_reason=sim_time_limit`).

Results write `agentx_c<concurrency>.json` plus `replay/{events.jsonl,requests.jsonl,cache_windows.jsonl,cache_configuration.json}`.
Reported metrics include output token throughput, time to first token (TTFT), inter-token latency (ITL) for multi-token outputs, reciprocal interactivity percentiles (`1 / pXX`), and GPU/CPU/computed prompt-source breakdowns.

## Observed runtime KV capacity

`--runtime_memory_profile PATH` overrides per-rank KV byte budgets on ordinary grouped-cache workers using measured post-initialization memory availability:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `deployment_identity` | Exact `worker_cache_budgets[].cache.memory_budget.deployment_identity` from a matching probe |
| `kv_bytes_per_rank` | Positive integer KV byte capacities in global rank order |
| `uncertainty_bytes` | Nonnegative rounding margin recorded in provenance |
| `provenance` | `{source, measurement_id, framework, framework_version, graph_strategy}` |

The allocator divides `kv_bytes_per_rank` by the physical slot stride and records `capacity_source: "runtime_observation"` alongside `analytical_available_kv_bytes`.
