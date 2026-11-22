# Device catalog (`data/devices/`)

Each YAML file in `data/devices/` defines one accelerator's compute peaks, memory hierarchy, interconnect links and analytical parameters.
The catalog loads device files recursively, including variants in subdirectories.

## Evidence grades and provenance

Every `SourcedValue` stores a numeric value, a `source_id` declared in the file's `sources:` mapping and an evidence grade.

| Grade | Meaning | Example |
| --- | --- | --- |
| **A** | Vendor datasheet, whitepaper or direct measurement on the target hardware | H100 HBM3 bandwidth 3.35 TB/s |
| **B** | Published third-party benchmark or complete-module measurement | MLPerf submission or imported module curve |
| **C** | Calibrated or derived estimate constrained by observations | Per-GPU NVL72 peak derived from rack specifications, or `calibrate` fit |
| **D** | Architecture-family analytical default | Default `gemm_mfu` for an uncalibrated device |

Results report the lowest grade among their active inputs.
`tokensim-operator-data calibrate` fits analytical efficiency parameters against measured tables and upgrades grade D defaults to grade C.

## Analytical parameters and memory reserves

When operator tables do not cover a query, the roofline model estimates execution time from floating-point operations (FLOPs) and memory traffic:

$$
t_{\mathrm{op}} = \max\left(\frac{F_{\mathrm{op}}}{P_d \eta_{\mathrm{op}}}, \frac{B_{\mathrm{op}}}{B_{\mathrm{mem}} \eta_{\mathrm{mem}}}\right) + \tau_{\mathrm{op}}.
$$

Parameters are grouped by device family (`nvidia_gpu`, `groq_tsp` and `generic`) and can be overridden per device under `analytical:`.
GEMM denotes matrix multiplication, MoE denotes mixture-of-experts computation and a collective is a multi-rank communication operation.

| Parameter | Meaning | `nvidia_gpu` default | `groq_tsp` default |
| --- | --- | --- | --- |
| `gemm_mfu` | Effective peak compute fraction for large GEMMs | 0.70 | 0.80 |
| `attention_mfu` | Effective peak compute fraction for attention cores | 0.45 | 0.60 |
| `moe_mfu` | Effective peak compute fraction for grouped/fused MoE | 0.50 | 0.70 |
| `memory_efficiency` | Effective memory bandwidth fraction | 0.85 | 0.90 |
| `elementwise_memory_efficiency` | Effective bandwidth fraction for elementwise kernels | 0.70 | 0.90 |
| `kernel_launch_us` | Fixed launch overhead per kernel in microseconds | 4.0 | 0.2 |
| `collective_launch_us` | Fixed launch overhead per collective in microseconds | 8.0 | 0.5 |
| `gemm_small_m_knee` | Row-count tile threshold below which GEMM compute scales down | 64.0 | 1.0 |

NVIDIA GPU catalog entries apply AIConfigurator's grade C system calibrations: `memory_efficiency: 0.8`, `kernel_launch_us: 3.0`, `collective_launch_us: 10.0` and `memory.reserved_bytes: 4169138176` (communication buffers plus runtime workspace).
`DeviceSpec.usable_memory_bytes` and `CacheConfig` deduct `memory.reserved_bytes` before sizing the attention key/value (KV) cache.

`DeviceSpec.peak_compute_for(dtype)` maps storage data types to compute pipelines: weight-only formats (`int8_wo`, `int4_wo`, `mxfp4_wo`) execute on `fp16`, `int4_a8` and `fp8_block` execute on `fp8`, and `mxfp4` executes on `nvfp4`.
When a device lacks a requested compute pipe, `COMPUTE_PIPE_FALLBACKS` tries `nvfp4 -> fp8 -> fp16 -> bf16`, `int4 -> int8 -> fp16 -> bf16`, `bf16 <-> fp16` and `tf32 -> fp32`.

## Bundled devices and operator packages

Measured operator packages are imported from NVIDIA AIConfigurator (`f254959eb89e2f206b8f9a77051644d7c1cbdb89`, Apache-2.0; see [`../operator_data/LICENSES.md`](../operator_data/LICENSES.md)) for `vllm` (default), `trtllm` and `sglang`.
Select a backend globally with `--operator_backend` or per worker group with `"operator_backend"`.

| `device_id` | Accelerator | Family | Spec grade | Bundled operator backends |
| --- | --- | --- | --- | --- |
| `a100_sxm_80g` | NVIDIA A100-SXM4-80GB | `nvidia_gpu` | A | `vllm`, `trtllm`, `sglang` |
| `a100_sxm_40g`, `a100_pcie_80g`, `a30` | NVIDIA Ampere GPUs | `nvidia_gpu` | A | analytical |
| `v100_sxm2` | NVIDIA V100-SXM2-32GB | `nvidia_gpu` | A | `nccl` collectives; analytical compute |
| `h100_sxm`, `h200_sxm` | NVIDIA H100 80GB / H200 141GB SXM | `nvidia_gpu` | A | `vllm`, `trtllm`, `sglang` (with DeepEP `ep_all2all`) |
| `h100_pcie`, `l4`, `rtx_4090` | NVIDIA Hopper / Ada PCIe GPUs | `nvidia_gpu` | A | analytical |
| `h20` | NVIDIA H20 96GB | `nvidia_gpu` | B | analytical |
| `l40s` | NVIDIA L40S | `nvidia_gpu` | A | `vllm`, `trtllm`, `sglang` |
| `rtx_pro_6000_server` | NVIDIA RTX PRO 6000 Blackwell Server | `nvidia_gpu` | C | `vllm`, `trtllm`, `sglang` |
| `b200_sxm`, `b300_sxm` | NVIDIA HGX B200 / B300 SXM | `nvidia_gpu` | A / C | `vllm`, `trtllm`, `sglang` (with DeepEP, MegaMoE, mHC) |
| `gb200`, `gb300` | NVIDIA GB200 / GB300 NVL72 | `nvidia_gpu` | C | `vllm`, `trtllm`, `sglang` (4 GPUs per tray) |
| `rubin` | NVIDIA Vera Rubin NVL72 | `nvidia_gpu` | C / D | analytical |
| `groqchip_v1` | GroqChip v1 LPU | `groq_tsp` | A | `analytical` package |
| `groq_lp30` | NVIDIA Groq 3 LP30 LPU (LPX256) | `groq_tsp` | A / C / D | analytical SRAM roofline / unit pipeline |
| `intel_arc_pro_b60` | Intel Arc Pro B60 XPU | `generic` | A / C | `vllm` (with oneCCL collectives) |

Upstream framework versions follow each hardware generation's latest complete profiling collection, backfilling unchanged shapes from prior versions and retaining `kernel_source` in the `kernel` column.
See [Operator latency and data packages](../../docs/reference/operator-latency.md) for lookup rules and `tokensim-operator-data` commands.

## Adding a device

1. Copy a close device YAML and set `device_id`, `display_name` and `aliases` (keep `B200`/`B300` for HGX parts and `GB200`/`GB300` for NVL72 superchips).
2. Populate dense `peak_compute` rates, `memory`, optional `on_chip_memory`, `interconnect` link IDs from `data/topologies/links.yaml` and `power`.
3. Declare every `source_id` under `sources:` with its evidence grade and citation.
4. Optionally import or collect operator tables under `data/operator_data/<device_id>/<backend>/` using [`scripts/collect/`](../../scripts/collect/README.md).

## High-bandwidth flash (HBF) GPU memory variants

The [`hbf/`](hbf/) subdirectory defines experimental memory variants (`h100_sxm_hbf1`, `h100_sxm_hbf1_48gib`, `h100_sxm_hbf2`, `h200_sxm_hbf1`, `h200_sxm_hbf2`, `b200_sxm_hbf1`, `b200_sxm_hbf1_48gib`, `b200_sxm_hbf2`) that pair standard GPU compute and interconnects with GDDR7 or HBM3e capacity and bandwidth profiles (grade D).
Select a variant explicitly in `worker_groups[*].hardware`, such as `data/clusters/8_h100/tp8_hbf1.json`.
For external flash KV storage behind a standard GPU cluster, use the [HBF storage plugin](../../examples/storage/hbf/README.md).
