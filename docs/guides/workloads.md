# Workloads and AgentX replay

`tokensim` accepts synthetic requests, length traces and AgentX conversation trees.
Set arrival rate and length distributions with CLI arguments or a JSON file passed to `--workload_config`.
The CLI values override corresponding file fields.
The serving configuration selects the cluster, scheduler and service time model independently of the workload.

| Workload type | Input |
| --- | --- |
| `synthetic` | Prompt and output lengths drawn from the configured distributions |
| `json_pairs` | A JSON array of `[input_length, output_length]` pairs with optional metadata as a third element |
| `qwen_jsonl` | One object per line with input and output lengths; may include timestamps and prefix identifiers |
| `agentx_weka` | AgentX JSON Lines traces of agent request trees, with timing and dependency metadata |

For `json_pairs` or `qwen_jsonl`, set `--dataset_path` and `--request_count`.
Use `--dataset_skip_count` to start at a later row.
To replay each selected record once, set `--request_count` to the number of rows you want to use after the skip.
The following example consumes each record in the three-row sample once.
Paths in this example are relative to a checkout.

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

See [prefix caching](prefix-cache.md) for request hashes and [the workload configuration example](../../README.md#workload) for latency targets.

## AgentX traces

AgentX replays main-agent and subagent requests with their dependencies.
Install the optional AgentX dependencies and download a pinned trace revision from the checkout.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agentx]'
hf download semianalysisai/cc-traces-weka-062126 traces.jsonl \
  --repo-type dataset \
  --revision 23f152f6f0f9399a85901b89a6458def0ef16729 \
  --local-dir dataset/agentx
echo '29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276  dataset/agentx/traces.jsonl' | sha256sum --check
```

The dataset is [SemiAnalysis's WEKA traces](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729) at revision `23f152f6f0f9399a85901b89a6458def0ef16729`.

Run the following example from the checkout with the trace downloaded above.

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

Set concurrency and the issue window for your workload.
The example uses GPU memory for KV storage.
See [KV storage](kv-storage.md) for rank-local offload and shared stores.
Select a model whose context limit and available KV budget can accommodate the chosen prompts.
Use a finite chunking budget with grouped KV; a speculative verification query needs at least one token plus the configured draft width.

## Replay and stopping

In `closed_loop` mode, a request is issued after its required predecessors finish and the recorded delay before the next request has elapsed.
Recorded API durations help determine request dependencies and delays.
The simulator calculates compute time from the selected model and latency backend, without adding those recorded API durations.
SPAWN identifies a subagent branch, and JOIN makes the parent wait for the required subagents to finish.
The simulator releases child work from the corresponding completion point in its replay graph.
Replay preserves these dependencies, the main agent's request sequences and independently running subagents.
Each replay of a trace tree has its own identifier that controls prefix cache sharing and stays the same across warmup and measurement.
When a tree finishes, its replay slot takes the next tree from the dataset, cycling back to the start when needed.

`--agentx_issue_mode recorded` schedules requests using the timestamps in each trace tree.
Each replacement tree uses its own recorded timestamps relative to its replay start.
A request waits longer if its dependencies have not finished by its scheduled time.
In `closed_loop` mode, idle limits shorten long gaps with no requests ready to run.
The results record how often these gaps are shortened and how much idle time is skipped.

`--sim_time` stops the simulation at the specified simulated time, including during warmup.
Pending transfers stop at that cutoff as well.
The saved partial results include `termination_reason=sim_time_limit` and metadata describing the replay state at the cutoff.
At the normal end of the measurement window, AgentX stops issuing new requests and allows outstanding requests to finish during `--agentx_grace_period`.
It then cancels the remaining requests, waits up to `--agentx_cancellation_timeout` for cancellation and pending transfers to finish, and checks cache references and allocation accounting.
An unacknowledged cancellation at the deadline stops the run with `termination_reason=failed`, including during warmup.
The `notdone` list and `failure.json` retain the IDs of outstanding requests.
If cleanup fails or the cache's allocation counts are inconsistent, the simulator saves results and diagnostic files before returning an error.

## Results and interpretation

The result directory contains `agentx_c<concurrency>.json` and a `replay/` directory with `events.jsonl`, `requests.jsonl`, `cache_windows.jsonl` and `cache_configuration.json`.
Failures also produce `failure.json` and `failed.txt` in that directory.
The configuration log stores cache descriptions that are shared by multiple log entries.

AgentX reports output tokens per second, request latency percentiles, time to first token (TTFT), inter-token latency (ITL), prefix-source counts, cache hit rates and replay state.
Prefix-source counts separate input tokens reused from GPU memory, loaded from CPU memory and computed by the model.
Output token totals count actual tokens generated by successful requests issued during the measurement window.
Root and subagent output totals and ITL summaries use the actual generated lengths, including when the model's context limit truncates an output.
Requests that generate only one token are excluded from AgentX ITL summaries.
Latency summaries retain finite positive observations, matching the InferenceX request-metrics parser.
Zero ITL observations do not enter either the latency or interactivity population, and each summary reports its own sample count.
Interactivity uses the inverse of the corresponding latency statistic, including the arithmetic mean.
For example, `interactivity_tps.p90` is `1 / tpot_s.p90`, and normalized interactivity uses end-to-end seconds per actual output token.
Percentiles use NumPy linear interpolation; computing percentiles of reciprocal samples would reverse the latency tail and change the interpolated value.
Results record `interactivity_percentile_semantics: inverse_of_matching_latency_percentile` to identify this convention.
Latency summaries exclude warmup, cancelled and unfinished requests.
Throughput divides successful output tokens by their observation span, including successful grace-period completions, so its denominator can exceed the issue window.
Engine statistics record repeated computation, time spent waiting for transfers and the measurement or formula used to estimate each operator's latency.

## Scheduled root arrivals

Use `--agentx_root_arrivals schedule.jsonl` with `--no-agentx_warmup` to start complete agent trees at explicit simulation times.
Each JSON Lines record contains a nonnegative `time_s` and a zero-based `trace_index` in the loaded corpus.
Provide the rows in nondecreasing time order.

```jsonl
{"time_s": 0.0, "trace_index": 0}
{"time_s": 0.25, "trace_index": 0}
```

The profiling window accepts new roots.
During grace, existing trees can issue ready turns without accepting new roots.
See [programmable KV cache policies](prefix-cache.md#programmable-cache-policies) for cache behavior during tool pauses.

## Observed runtime KV capacity

`--runtime_memory_profile PATH` supplies per-rank KV bytes measured after a serving stack reserves weights, graphs and workspace.
The profile applies to ordinary workers with grouped compressed caches.
To build it, run a short probe with the intended deployment settings, then copy `worker_cache_budgets[].cache.memory_budget.deployment_identity` from the result.
The identity covers model, hardware, layout, scheduler and connector settings.
Use a measurement from that deployment and record the source.
[InferenceX reference 440239](https://inferencex.semianalysis.com/agentx/telemetry?id=440239) is an example of the telemetry a source can provide.

| Profile field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `deployment_identity` | Exact identity exported by the matching simulation configuration |
| `kv_bytes_per_rank` | Positive integer byte capacities in global rank order, with one entry for every rank |
| `uncertainty_bytes` | Nonnegative measurement rounding uncertainty, retained in provenance without changing the nominal capacity |
| `provenance.source` | Measurement artifact or source URL |
| `provenance.measurement_id` | Immutable deployment/run identifier |
| `provenance.framework` | Serving framework name |
| `provenance.framework_version` | Exact source framework version or commit |
| `provenance.graph_strategy` | Source CUDA graph policy and capture configuration |

Unknown profile fields, missing provenance, unmatched identities, incomplete rank coverage, capacities exceeding the device budget with weights, and capacities smaller than one physical KV slot cause an error.
The available bytes are divided by the existing physical slot size; allocation continues to use the existing grouped cache manager.
The result retains analytical deductions and `analytical_available_kv_bytes` for comparison, and marks the selected `capacity_source` as `runtime_observation`.
It records the profile hash, rank, provenance, uncertainty, and `deductions_applied_to_observation: false` in `runtime_observation`.
Recorded bytes already include runtime reservations and receive no additional graph/workspace deduction.
Capacities inferred from one rank require an explicit uniform-rank assumption in provenance.
