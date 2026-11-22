# Model configurations and architectures

Catalog entries in `data/models/*.yaml` define transformer dimensions, quantization formats, attention geometry, mixture-of-experts (MoE) routing and speculative decoding.
Pass a catalog ID or YAML path to `--model`.

## Mixture of experts (MoE)

Configure MoE layers under `moe:` in the model YAML:

| Field | Default | Meaning |
| --- | --- | --- |
| `num_experts`, `num_experts_per_tok` | required, `1` | Total routed experts and selected experts per token (`top_k`) |
| `moe_intermediate_size` | `intermediate_size` | Hidden width per routed expert |
| `num_shared_experts` | `0` | Dense shared-expert width multiplier (`num_shared_experts * moe_intermediate_size`) |
| `num_moe_layers`, `first_k_dense_replace`, `moe_layer_freq` | `num_layers`, `0`, `1` | MoE layer count, starting layer index and stride |
| `routing` | `uniform` | Synthetic routing (`uniform`, `hot`, `skew`, `burst` or `{distribution, hot_experts, hot_expert_fraction}`) |

```bash
tokensim \
  --batching paged-attn \
  --model mixtral-8x7b \
  --cluster ./data/clusters/8_h100/moe_ep8_deepep.json \
  --request_count 16 \
  --prefill_mean_len 512 \
  --prefill_range_len 0 \
  --decode_mean_len 32 \
  --decode_range_len 0 \
  --qps 4 \
  --verbose none
```

Trace records can supply `expert_histogram`, `prefill_expert_histogram` or `decode_expert_histogram` (as `{expert_id: count}`, `[expert_id, ...]` or `[[expert_id, count], ...]`), or token-level `routing_assignments`.
Without trace histograms, TokenSim generates deterministic seeded assignments; skewed distributions prefer measured `power_law_1.2` MoE rows when available and otherwise add rank-imbalance straggler time.
With `--moe_backend deep_gemm_amxf4_mega_moe`, single-node PP1 EP runs (with `--dp_lockstep` when DP > 1) price dispatch, routed experts and combine from the bound `dsv4_megamoe` table.

Results report `effective_moe_config`, `moe_expert_placement`, `moe_per_expert_load`, `moe_per_rank_expert_load`, `moe_expert_load_imbalance_ratio`, `moe_compute_latency`, `moe_straggler_latency` and `parallel_ep_all2all_latency`.

## DeepSeek-V3 multi-head latent attention (MLA)

`data/models/deepseek-v3-671b-mla.yaml` models DeepSeek-V3 (61 layers: 3 dense + 58 MoE, hidden size 7168, 128 query heads, query rank 1536, KV latent rank 512, content head dim 128, RoPE dim 64, value dim 128).
Each token caches one shared 512-element latent vector plus one 64-element RoPE key vector per layer ($d_{\mathrm{cache}} = 576$, or 1,152 bytes/token/layer at BF16), replicated across TP ranks that own the layer while query heads and projections shard by TP.

- **Decode (`absorbed`)**: compresses queries, projects latent KV + RoPE, absorbs key expansion into per-head queries (`batched_gemm`), evaluates attention over the 576-wide compressed cache (`mla_attention`) and expands latent values (`batched_gemm`) before output projection (`gemm`).
- **Prefill (`expanded`)**: expands latent KV into local heads before causal attention while persisting the compressed 576-wide cache.

## DeepSeek-V4 and speculative decoding

`deepseek-v4-pro` combines compressed/sliding sparse attention (CSA/HCA), indexer and compressor state, hyper-connection residual mixing (`mHC`) and packed MoE layers.
`deepseek-v4-pro-0813-dspark` adds a three-layer DSpark speculative draft model with sequential Markov heads for TP1 serving.

Configure multi-token prediction (MTP) draft verification with `--mtp_num_speculative_tokens K`:

- `--mtp_acceptance_length`: two-point approximation matching the target mean accepted tokens per step.
- `--mtp_acceptance`: JSON object specifying per-position `conditional` probabilities or an `empirical` distribution over accepted lengths `1..K+1` (with `--mtp_acceptance_length 1`).

Acceptance lengths sample deterministically per `(seed, request_id, decode_round)` and report under `module_stats.speculative.acceptance`.

## Gemma-4 hybrid layers and Groq LP30 / Rubin SRAM modeling

`data/models/gemma-4-31b-text.yaml` models Gemma 4 31B's 60 text layers (five 1,024-token sliding-window layers with 16 KV heads of dim 256 followed by one global-attention layer with 4 shared-K/V heads of dim 512, plus `block_layout` norms, scales and logit softcapping).

For ordinary per-layer attention geometries, rank $r$'s logical KV footprint for context $L$ is:

$$
M_{\mathrm{KV},r} = \sum_{\ell \in \mathcal{L}_r} 2 b_{\mathrm{KV}} H_{\mathrm{KV},r,\ell} d_\ell \min(L, w_\ell).
$$

On `groq_lp30` (`data/devices/groq_lp30.yaml`, 500 MB on-chip SRAM at 150 TB/s and 1.25 TB/s per-direction injection per chip across 256 chips in `groq_lpx256`), admission enforces strict on-chip SRAM residency for weights, protocol workspaces (`analytical.lpx_memory`) and windowed KV state; `rubin.yaml` and `rubin_nvl72.yaml` provide the companion Vera Rubin NVL72 catalog.

Run a fixed-batch colocated LPX256 decode sweep with checkpoint reuse:

```bash
python scripts/run_lpx.py \
  --model gemma-4-31b-text \
  --device groq_lp30 \
  --tp 16 \
  --pp 16 \
  --dtype fp8 \
  --kv-cache-dtype fp8 \
  --batches 1 4 \
  --context 1024 \
  --steps 2 \
  --output outputs/lpx
```
