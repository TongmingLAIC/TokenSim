# Operator tracing and profiler comparison

## Global operator tracing

Enable timeline capture across colocated serving, prefill/decode (PD) disaggregation and attention/FFN (AF) disaggregation on the main CLI or through `SimulationConfig(trace=TraceConfig(...))`:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--trace_mode` | `off` | `off` disables capture, `validate` checks causal and resource invariants online, and `full` exports a timeline |
| `--trace_output` | `<results_path>/<stem>.trace.json` | Output trace path |
| `--trace_config` | unset | Optional JSON file with `TraceConfig` selectors (`format`, `workers`, `phases`, `iterations`, `layers`, `event_types`, `detail`, `fields`) |

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

Chrome Trace JSON exports microsecond timestamps and durations (retaining simulation seconds in `args`) for inspection in [Perfetto](https://ui.perfetto.dev/).
`format` supports `json`, `archive` and `msgpack` (with `tokensim[trace]`); layer `-1` denotes model-boundary operations (embedding, final norm, LM head, sampling).

## PyTorch Profiler comparison

`TokenSim.profiler.compare` aligns a simulated trace against PyTorch Profiler captures (`--collection manifest.json` from [`scripts/collect/collect_operator_trace.py`](../../scripts/collect/README.md#operator-timeline-collection) or explicit `--profile` files):

```bash
python -m TokenSim.profiler.compare \
  --simulated outputs/operator-trace/simulated.trace.json \
  --collection outputs/profiler/qwen3-8b/manifest.json \
  --output outputs/operator-trace/comparison
```

Order-preserving matching pairs simulated and observed spans per `(rank, forward, layer)` by operation and shape, grouping fused residual/norm or reduction kernels when observed or declared via `--mapping` (`{schema_version: 1, aliases, rank_map, groups}`).
Outputs write `comparison.json` (coverage, module-span and device-busy duration deltas, peer-wait attribution on shared clocks), `operators.csv` and `comparison.trace.json`.

## Paired timeline figures

Render aligned simulated-vs-profiled SVG/PNG timelines and a `.timeline.json` geometry sidecar with `tokensim[plot]`:

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

Select ranges and styling with `--forward`, `--layer`, `--rank`, `--detail` (`kernel` or `module`), `--alignment` (`global` or `selection`), `--start-us`, `--end-us`, `--formats` (`png`, `svg`), `--width`, `--dpi`, `--title`, `--simulated-label` and `--profiled-label`.
