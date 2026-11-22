# Operator latency and data packages

TokenSim estimates operator and communication service times by combining measured operator/module tables with hardware roofline and interconnect models.

## Lookup and fallback policies

Select `--latency_backend operator_table` (default) or `analytical`, and choose `--latency_fallback`:

| Fallback | Behavior |
| --- | --- |
| `table_first` (default) | Query measured tables; fall back to the analytical model on misses and record them in `missing_shapes_<qps>.json` |
| `table_only` | Require measured coverage for compute and communication queries, raising `MissingOperatorDataError` on misses |
| `analytical_only` | Evaluate every query with analytical formulas (implied by `--latency_backend analytical`) |

Worker-group `operator_backend` overrides `--operator_backend`, which defaults to an available catalog package (`vllm` preferred).
`--decode_context_bucket` sets decode context rounding (default 128 tokens).
`--attention_backend module`, `--mhc_backend module` and `--moe_backend deep_gemm_amxf4_mega_moe` bind complete measured modules declared under the model's `operator_modules`, skipping constituent primitive operations covered by those modules.

Table lookup checks exact normalized discrete keys, interpolates recursively across numeric axes (for example `k -> n -> m` on GEMM) in linear, logarithmic or `analytical` work space, and extrapolates from the nearest boundary anchor via `measured(boundary) * analytical(target) / analytical(boundary)` within per-axis ratio bounds (16x default, 64x for attention sequence axes).

## Analytical compute and LP30 unit models

The roofline estimator computes `max(compute_time, memory_time) + kernel_launch_us` using device peaks and family or per-device efficiency parameters (see [Device catalog](../../data/devices/README.md#analytical-parameters-and-memory-reserves)), with optional per-table linear calibration `multiplier * roofline_us + offset_us`.

For Groq LP30 / LPX architectures (`groq_lp30`), the default `groq_tsp` roofline uses on-chip static random-access memory (SRAM) bandwidth (`150 TB/s * 0.9 = 135 TB/s`) plus vector unpacking for packed weights.
Custom device YAML files can also set `analytical.lpx_pipeline` to separate matrix, vector, exponential, shuffle and SRAM unit times across `N` tiles:

$$
t_{\mathrm{tiled}} = \max_j t_j + \frac{\sum_j t_j - \max_j t_j}{N} + \tau_{\mathrm{op}} + \tau_{\mathrm{fill}}.
$$

## Communication models

Hierarchical collectives evaluate `ring`, `tree` or `direct` schedules at each crossed topology level using link bandwidth, latency, endpoint injection limits, efficiency and oversubscription, consulting measured `collective` or `ep_all2all` tables when available.
`CollectiveQuery(schedule="global_ring", algorithm="ring")` models a single ordered ring across all ranks under `analytical_only`, accounting for shared NIC/rack injection contention under non-contiguous rank placement.
Flat AF collectives compute $t_{\mathrm{comm}} = n_{\mathrm{rounds}}\alpha + S_{\mathrm{wire,rank}} / B_{\mathrm{direction}}$.

## Operator data packages and CLI

Bundled packages under `data/operator_data/<device>/<backend>/` store Parquet tables (`latency_us`, `source_id`) and `generation_meta.yaml` imported from [NVIDIA AIConfigurator](https://github.com/ai-dynamo/aiconfigurator/tree/f254959eb89e2f206b8f9a77051644d7c1cbdb89) (`f254959eb89e2f206b8f9a77051644d7c1cbdb89`).

```bash
tokensim-operator-data import-aiconfigurator \
  --system-dir /tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/b200_sxm \
  --device b200_sxm \
  --backend vllm \
  --upstream-commit f254959eb89e2f206b8f9a77051644d7c1cbdb89 \
  --module-source dsv4_megamoe=/tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/gb200 \
  --out /tmp/b200-vllm
tokensim-operator-data validate /tmp/b200-vllm
```

| Subcommand | Output |
| --- | --- |
| `manifest` | Required operator keys across selected models, TP sizes and token grids |
| `generate` | Analytical Parquet tables and work breakdowns for a device |
| `import-aiconfigurator` | Imported compute, collective, DeepEP and module tables (`--gpus-per-node` for flat wide-EP logs) |
| `import-nccl` | Multi-node or custom `nccl-tests` curves merged into a package |
| `validate` | Schema, key uniqueness, row-count and provenance verification |
| `coverage` | Exact, interpolated, extrapolated and missing shape counts for a deployment |
| `calibrate` | Fitted analytical efficiency parameters with holdout error statistics |

## Result provenance fields

| Result field | Description |
| --- | --- |
| `latency_backends` | Active backend, fallback policy, dataset version, row counts and module bindings per device |
| `operator_query_count`, `operator_match_type_counts`, `operator_table_match_counts` | Query totals by match type (`exact`, `interpolated`, `extrapolated`, `analytical`) overall and per table |
| `operator_component_seconds` | Simulated seconds across `gemm`, `attention`, `moe`, `elementwise`, `lm_head`, `comm` and `overhead` |
| `operator_missing_shape_count`, `operator_missing_per_table`, `operator_missing_shapes` | Unanswered table queries grouped by `(table, discrete_key)` |
| `module_stats` | Match counts and component breakdown for bound operator modules |
| `parallel_comm_match_type_counts`, `af_operator_stats` | Communication and AF stage provenance summaries |
