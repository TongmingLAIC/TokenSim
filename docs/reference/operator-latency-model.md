# Operator latency model

TokenSim estimates scheduler service times from measured operators, complete module tables and analytical models.
A device supplies peak compute and memory rates; a topology supplies link rates and propagation delays.
A [model configuration](models.md) supplies layer geometry and bindings for fused operator modules.
[Operator data packages](operator-data.md) identify the device, backend, measurement source and timing boundary for each estimate.

## Select timing inputs

The default `operator_table` backend tries measured tables with `table_first` fallback.
A worker group's explicit `operator_backend` takes precedence over the CLI option.
Without either setting, the catalog chooses an available package for the device.
A package has latency in microseconds and a `source_id` for each row.
Discrete fields such as data type, group size or kernel scope must match before numeric axes can be interpolated.

| Fallback | Behavior |
| --- | --- |
| `table_first` (default) | Table lookup first; on failure, use the analytical model **and record the miss** in the worker's `MissingShapeReport` |
| `table_only` | Table lookup only; the first miss, including a communication query without a measured row, raises `MissingOperatorDataError` and stops the run |
| `analytical_only` | Skip tables entirely (`--latency_backend analytical` implies this) |

`--decode_context_bucket` controls the context bucket width used by decode lookup (128 tokens by default).
Select `--attention_backend module` or `--mhc_backend module` to use a model-bound module.
`--moe_backend deep_gemm_amxf4_mega_moe` selects the bound fused expert table.
A complete module omits separate estimates for operations included in its measurement.
A missing compatible row for a configured measured module can fail the run even with `table_first`; choose compatible data or an analytical configuration.
Ragged attention batches account for each request's context and newly processed tokens.
The analytical kernel cost and the cache page capacity are separate quantities.

## Lookup and fallback

The lookup tests a normalized exact key, interpolates between compatible anchors, then scales the nearest boundary anchor within its allowed extrapolation range.
It resolves numeric axes recursively in the table's declared order and keeps previously selected anchor coordinates bound.
A GEMM query resolves `k`, then `n`, then `m`.
For each axis, interpolation uses the declared linear or logarithmic coordinate.
For Python callers, `LookupPolicy(interpolation_space="analytical")` uses `(A(target) - A(lower)) / (A(upper) - A(lower))` as the interpolation weight when the supplied analytical scaler gives finite increasing coordinates.
Otherwise it records `axis_interpolation_fallback` and uses the declared axis.

Analytical extrapolation uses `measured(boundary) * analytical(target) / analytical(boundary)`.
The default limit is 16× from the boundary per axis; some attention context axes allow 64× and DeepEP defines separate limits for expert count and top-k.
`LookupPolicy` also supports empirical power-law growth, dimension scaling, holding the boundary estimate or disabling extrapolation.
The default bound-axis convention is reported as `lookup_semantics: bound_axes_v1`.

## Analytical service

The device-family roofline model takes the larger of compute time and memory service time, then adds the launch cost.
Compute time divides operation count by effective peak compute; memory time divides bytes by effective bandwidth.
An optional per-table calibration applies `multiplier * roofline + offset_us` to service in microseconds.
Small GEMM shapes can use an effective row-count floor to model tile utilization.
The Groq Tensor Streaming Processor uses on-chip memory rates and reports whether local weights fit.
See [LP30 and Rubin](lpx.md) for the native rack-scale model.

<a id="analytical-defaults"></a>

### Device parameters and provenance

The [hardware schema](../../TokenSim/hardware/README.md) lists family defaults and their evidence grades.
A device can override them in `analytical:`, with a `source_id` and `grade`.
For example, device files can set `memory_efficiency`, `kernel_launch_us` and `collective_launch_us` from AIConfigurator system metadata.
The cache budget deducts `memory.reserved_bytes` separately for communication buffers and workspace.

<a id="gb300-derivation"></a>

The [GB200](../../data/devices/gb200.yaml) and [GB300](../../data/devices/gb300.yaml) catalogs derive per-GPU dense peaks from NVL72 rack totals (rack sparse total / 72 / 2, with FP4 from the dense rack figure).
The catalogs retain the sources and mark these derived values as grade C.

## Communication model

The hierarchical model chooses a ring, tree or direct algorithm at each topology level.
Each level accounts for propagation and serialized bytes using the configured link, endpoint rate, efficiency and oversubscription.
A matched communication table can supply collective time according to the selected fallback.
The ordinary collective table matches operation, activation dtype, group size, node count and message bytes.
The `ep_all2all` table matches the dispatch or combine phase, DeepEP mode, rank and expert geometry, and tokens.

A Python query can select `CollectiveQuery(schedule="global_ring", algorithm="ring")` to model one ordered ring across the full communicator under `analytical_only`.
Each successor edge uses its lowest common topology level.
Concurrent edges share the appropriate node or rack injection budget per direction, so interleaved rank order can cost more than contiguous placement.
The analytical schedule supports all-reduce, all-gather and reduce-scatter.
All-gather takes the complete gathered buffer size; `flat_collective` uses a local input size.
Use analytical service for explicit schedule studies: the current collective table key has no algorithm or schedule field and a matched curve can take precedence over the formula.

## Interpret estimates

| Field | Description |
| --- | --- |
| `latency_backends` | Per device: backend, fallback policy, dataset version, operator backend, table row counts, analytical model, cost scopes and module descriptions |
| `operator_query_count` | Number of operator queries |
| `operator_match_type_counts` | Counts per match type across all operator queries |
| `operator_table_match_counts` | The same counts per table |
| `operator_component_seconds` | Simulated seconds per component: `gemm`, `attention`, `moe`, `elementwise`, `lm_head`, `comm`, `overhead` |
| `operator_missing_shape_count` | Unique shapes (compute and communication) that no table answered |
| `operator_missing_shape_groups` | Number of `(table, discrete key)` groups those shapes fall into |
| `operator_missing_per_table` | Per table: missing groups, shapes and recorded misses |
| `operator_missing_shapes` | The first 200 groups (format below) |
| `module_stats` | Per operator module (when modules are active): match counts and time composition |
| `parallel_groups` | TP/EP group of rank 0 and the topology level each spans (`ep_group_size`, `ep_group_link`, ...) |
| `parallel_link_type_counts` | Communication event counts per topology level |
| `parallel_comm_match_type_counts` | Communication estimates by match type (measured, interpolated, extrapolated, analytical) |
| `af_latency_providers` | AF runs only: which compute / transfer provider answered (`operator_table` + `topology_links`, or the injected test providers) |
| `af_operator_stats` | AF runs only: the operator match-type counts and missing shapes behind the AF stage service times |

`missing_shapes_<qps>.json` groups unanswered estimates by table and discrete key, records axis ranges and distinguishes absent tables, incompatible discrete keys and out-of-range shapes.
The data package records which costs are direct measurements, derived estimates or formulas.
Use the [tracing guide](operator-tracing.md) for per-operation timelines and profiler comparisons.
