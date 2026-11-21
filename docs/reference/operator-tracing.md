# Operator tracing and Profiler comparison

## Global operator tracing

The main simulation CLI supports the same operator tracing configuration for colocated serving, prefill/decode (PD) disaggregation and attention/feed-forward (AF) disaggregation.
Tracing defaults to `off`.
`SimulationConfig(trace=TraceConfig(...))` provides the corresponding Python interface through `TokenSim.trace.config`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--trace_mode` | `off` | `off` skips recording, `validate` checks and releases completed events, and `full` exports an operator timeline. |
| `--trace_output` | `<results_path>/<result_stem>.trace.json` | Destination for a full trace; distinct result names receive distinct trace files, and binary formats use the corresponding suffix. |
| `--trace_config` | unset | JSON file containing global trace settings; explicitly supplied CLI values override its corresponding fields. |

```bash
tokensim \
  --model llama-3-8b \
  --latency_backend analytical \
  --request_count 2 \
  --prefill_mean_len 32 \
  --decode_mean_len 3 \
  --trace_mode full \
  --trace_output outputs/operator-trace/simulated.trace.json \
  --results_path outputs/operator-trace
```

The result JSON contains a `trace` descriptor with the output path, schema version and completeness.
Chrome Trace JSON uses microseconds for timeline timestamps and durations, while event arguments retain simulation seconds.
Open the exported JSON in [Perfetto](https://ui.perfetto.dev/) to inspect per-layer operations, rank tracks and dependency arrows.
An interrupted run exports completed work and observed transfer intervals, with incomplete capture identified in metadata.
Synchronous connector records describe consumed blocking invocations, and queued storage transfers expose actual resource admission and service intervals.
Shared asynchronous links identify resource-only ownership when request correlation is unavailable.

`TraceConfig` also accepts `validate`, `format`, `workers`, `phases`, `iterations`, `layers`, `event_types`, `detail` and `fields`.
`format` selects `json`, `msgpack` or `archive`; MessagePack requires the optional `tokensim[trace]` dependencies.
Worker, phase, iteration, layer and event-type selectors filter exported events without changing execution or removing records required for validation.
Layer indices are global model indices, and `-1` represents model-boundary work.
`detail="timing"` exports common timing arguments, while `fields` selects named event payload fields.
Dependency arrows connect selected endpoints only.
Direct JSON exports index dependency endpoints on disk and stream event records.
The fixed-batch [LPX command](lpx.md#run-and-interpret-results) defaults to full recording and accepts the same trace modes.
Explicit output paths and export selectors require `full` mode.

Operator costs are evaluated once and expanded into their model execution positions when recording is enabled.
Fused module estimates retain their measured boundaries, and shared AF executions contribute physical service once.
Ordinary tensor-parallel tracks that reuse a representative estimate identify the symmetric-rank assumption in their event details.
Tracing observes the selected simulated schedule.
See [event ownership and collection](#event-ownership-and-collection) for execution and comparison boundaries.

## PyTorch Profiler comparison

Use `python -m TokenSim.profiler.compare` to compare a simulated trace with one or more PyTorch Profiler JSON or gzip-compressed JSON files.
The offline tool requires no GPU, PyTorch or vLLM installation.

```bash
python -m TokenSim.profiler.compare \
  --simulated outputs/operator-trace/simulated.trace.json \
  --collection outputs/profiler/qwen3-8b/manifest.json \
  --output outputs/operator-trace/comparison
```

Central processing unit (CPU) ranges, graphics processing unit (GPU) activity and launch correlation establish operator ownership.
The [collection environment](#collection-environment) and [collector guide](../../scripts/collect/README.md#operator-timeline-collection) describe the paired structure/timing protocol and source verification.
`--collection` selects one completed collection's timing sample; `--profile` accepts explicit external trace files.
The importer distinguishes CPU annotations from their GPU counterparts so each forward is counted once.
Lossy simulated exports must retain `node_id`, `worker_id`, `layer_id` and `iteration_id` for comparison.
Full detail retains shape and provenance evidence; timing detail also retains the warmup/profiling distinction.

Matching uses rank, forward occurrence, layer, compatible shapes and operation order.
Forward token counts and an operator's flattened tensor rows are different shape fields.
Order-preserving matching accepts mandatory pairs and retains ambiguity for occurrences without a unique correspondence.
Observed fused residual/normalization kernels and projection or MoE modules containing a reduction can group their corresponding simulated operations, with source member identifiers retained.
Ambiguous and unmatched operations remain visible.
Incomplete simulated service is excluded from duration comparisons.

`--mapping` accepts an optional JSON file with version 1 of this schema:

```json
{
  "schema_version": 1,
  "aliases": {
    "custom_attention": "attention"
  },
  "rank_map": {
    "0": "2"
  },
  "groups": [
    {
      "simulated": ["attention_residual", "ffn_norm"],
      "profiled": "ffn_norm"
    }
  ]
}
```

Alias keys use lowercase operation names.
`rank_map` maps profiled rank identifiers to simulated worker identifiers, independently of `CUDA_VISIBLE_DEVICES` numbering.
Explicit groups combine contiguous simulated operations within a layer and forward.
The known residual/normalization adapter can also group a previous layer's residual with the following layer's input normalization when observed kernel evidence establishes that fusion.

| Artifact | Contents |
| --- | --- |
| `comparison.json` | Input hashes, available identity checks, matching evidence, coverage, timing errors and per-rank/per-clock accounting. |
| `operators.csv` | Matched, unmatched, ambiguous and excluded comparison units. |
| `comparison.trace.json` | Simulated and observed tracks with translated time origins and original timestamps in event arguments. |

The legacy `service_error_us` and relative-error columns compare module invocation spans.
The additional `simulated_device_service_us`, `profiled_device_service_us` and `device_service_error_us` columns compare modeled service with the union of observed device-work intervals.
Aggregate adjusted service is null if any matched scope has unresolved waiting; `comparable_profiled_device_service_us` explicitly sums the comparable subset.
Known waiting totals are named `known_matched_profiled_peer_wait_us` and `known_collective_peer_wait_us`, with unresolved calls retained in diagnostics.
`profiled_internal_gap_us` keeps module gaps separate.
`profiled_collective_residence_us` retains full collective kernel residence.
For the inspected vLLM custom all-reduce kernels with an entry barrier, `profiled_peer_wait_us` records the interval until the last corresponding peer starts.
The remaining device-work intervals retain any computation overlapping that wait.
This attribution requires complete participant groups and a shared host, boot identity, time namespace and Kineto timestamp basis recorded by the collector.
NCCL calls and other kernels without a proven entry-barrier contract retain unknown peer wait and unknown adjusted service.
Unknown clocks, missing peers and nonoverlapping collective intervals also produce null adjusted-service fields with an explicit reason.
After-arrival residence includes kernel and barrier overhead as well as device communication service.
The combined timeline preserves original spans and adds `collective_peer_wait` intervals.
Zero observed duration produces a null relative error with a reason.
A module invocation span runs from its earliest attributed GPU start through its latest attributed GPU completion and includes internal gaps.
Kernel-duration sum and device busy-time union are reported separately.
Rank-summed spans measure cumulative device time across ranks.
Inputs with a proven shared clock use one common translation so rank-arrival differences remain visible.
Other inputs are translated independently at their earliest unambiguous matches, and cross-rank overlap remains unavailable.
The CLI returns status 2 when no operators match and reports invalid inputs as argument errors.
This fixed-batch comparison measures ownership, order and accounting for the selected workload.
### Paired timeline images

The offline image exporter produces paired timelines in the layout of [Charon Figure 8](https://arxiv.org/pdf/2605.17164#page=10), with simulated operators above observed GPU activities.
Install the optional plotting dependency and select a layer from an existing comparison report:

```bash
python -m pip install -e '.[plot]'
python -m TokenSim.profiler.plot \
  --comparison outputs/operator-trace/comparison/comparison.json \
  --output outputs/operator-trace/figures/prefill-layer0 \
  --forward 0 \
  --layer 0 \
  --rank 0 \
  --alignment selection
```

The output stem receives `.png`, `.svg` and `.timeline.json` suffixes.
The JSON sidecar retains full labels, original timestamps, display translations, selectors and the source report hash.
Long names are truncated to fit inside visible bars, and intervals too narrow for text retain their complete names in the sidecar.
Scalable Vector Graphics (SVG) labels remain editable text.
Portable Network Graphics (PNG) exports default to 180 dots per inch.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--forward` | all | Forward occurrence, starting at zero. |
| `--layer` | all | Global layer index, or `-1` for model-boundary work. |
| `--rank` | all | Native rank identifier; repeat to select several ranks. |
| `--detail` | `kernel` | Show simulated component intervals and observed kernels/copies; `module` shows module envelopes with device activity overlaid. |
| `--alignment` | `global` | Retain comparison clock translations; `selection` translates each side's selected origin to zero. |
| `--start-us` | selected minimum | Left display boundary in microseconds after alignment. |
| `--end-us` | selected maximum | Right display boundary in microseconds after alignment. |
| `--formats` | `png svg` | Image formats to write. |
| `--width` | `20` | Figure width in inches. |
| `--dpi` | `180` | PNG resolution, from 72 to 600 dots per inch. |
| `--title` | generated | Figure title. |

Both panels use the same time scale and retain execution gaps and collective kernel residence.
Known peer waits appear in red without shortening the original kernels.
Distinct streams occupy separate rows, with additional rows for overlapping intervals.
Legacy reports without per-activity stream metadata use a row labeled `stream unavailable`.
Regenerate the comparison from its original captures to retain stream metadata.
Clocks without common-clock evidence remain independently aligned and carry a warning about unavailable cross-rank overlap.
Selectors apply to each side's own forward, layer and native rank fields, including unmatched and excluded work when those fields are available.
Selection-relative alignment preserves inter-rank offsets for ranks with a verified shared clock.
The optional image exporter uses Matplotlib.

## Event ownership and collection

Recording attributes work to physical resources where they are known.
An incomplete operation reports the observed interval and completion status.
A wait can be attributed to dependencies, buffering or a resource queue; it is not automatically device service.
Filtered exports emit dependency arrows for visible endpoint pairs.

<a id="collection-environment"></a>

The [collection guide](../../scripts/collect/README.md#operator-timeline-collection) documents the GPU-side vLLM collector.
The collector labels ownership in a structure pass and captures GPU time in separate timing passes.
Its manifest pins capture hashes and records the actual selected timing sample, unprofiled baselines, framework and model configuration.
Timing and structure passes must agree on the observed activity inventory for ownership to transfer.
The offline comparison tools can run without a GPU.
