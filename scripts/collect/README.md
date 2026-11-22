# Hardware latency and timeline collection (`scripts/collect/`)

These scripts profile operator tables and execution timelines on target GPU nodes for `data/operator_data/<device_id>/<backend>/` and `TokenSim.profiler.compare`.

| Script | Output | Requirements |
| --- | --- | --- |
| `run_aiconfigurator.sh` | `gemm`, `context_attention`, `generation_attention`, `moe`, single-node `collective` and module Parquet tables | Target framework (`vllm`, `trtllm` or `sglang`), `nccl-tests`, `mpirun` |
| `collect_elementwise.py` | `elementwise` (`rmsnorm`, `rope`, `residual_add`, `swiglu`) merged into a package | PyTorch with CUDA; vLLM for `--kernel vllm` |
| `run_nccl_tests.sh` | Multi-node or fabric-specific `collective` curves merged into a package | `nccl-tests`, `mpirun` or `srun` |
| `collect_operator_trace.py` | Paired structure and GPU-only PyTorch Profiler traces plus `manifest.json` | vLLM with `profiler_config`, `worker_extension_cls` and `collective_rpc` |

## Preparation

1. Lock GPU clocks (`sudo nvidia-smi -pm 1` and `sudo nvidia-smi -ac <mem>,<sm>`) or record active clocks with `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv`.
2. Ensure `data/devices/<device_id>.yaml` exists (see [`data/devices/README.md`](../../data/devices/README.md#adding-a-device)) and install TokenSim (`python -m pip install -e .`).
3. Optionally generate a workload shape manifest for `collect_elementwise.py` and post-collection coverage checks:

```bash
tokensim-operator-data manifest \
  --models llama-3-8b,llama-3-70b,mixtral-8x7b \
  --tp 1,2,4,8 \
  --out tmp/manifests/llama3-mixtral.yaml
```

## 1. Compute operators and single-node collectives (`run_aiconfigurator.sh`)

```bash
DEVICE=rtx_4090 \
BACKEND=vllm \
  ./scripts/collect/run_aiconfigurator.sh
```

The script checks out AIConfigurator at `AIC_COMMIT` (default `f254959eb89e2f206b8f9a77051644d7c1cbdb89`), runs `collector/collect.py --resume` into `RUN_DIR` (`./tmp/collect/<DEVICE>/<BACKEND>`), runs single-node `collect_comm.sh` unless `SKIP_COMM=1`, and imports and validates `PACKAGE` (`./data/operator_data/<DEVICE>/<BACKEND>`).
Key environment variables are `DEVICE`, `BACKEND` (`trtllm`, `vllm`, `sglang`), `OPS`, `MODEL_PATH`, `GPU_TYPE`, `GPUS_PER_NODE`, `SKIP_COMM`, `RUN_DIR` and `PACKAGE`; pass `--import-only` to import an existing `RUN_DIR`.

## 2. Elementwise operators (`collect_elementwise.py`)

```bash
python scripts/collect/collect_elementwise.py \
  --manifest tmp/manifests/llama3-mixtral.yaml \
  --device rtx_4090 \
  --package data/operator_data/rtx_4090/vllm \
  --backend vllm \
  --kernel vllm \
  --operator "$USER"
```

`--kernel vllm` times vLLM custom ops (`rms_norm`, `rotary_embedding`, `silu_and_mul`), while `--kernel eager` times PyTorch eager kernels, merging median CUDA event latencies into `--package`.

## 3. Multi-node collectives (`run_nccl_tests.sh`)

```bash
DEVICE=h100_sxm \
BACKEND=vllm \
DTYPE=bfloat16 \
NCCL_TESTS=/opt/nccl-tests/build \
  LAUNCHER="mpirun -np 16 -H nodeA:8,nodeB:8 -x NCCL_DEBUG=INFO" \
  ./scripts/collect/run_nccl_tests.sh 8 2
```

Positional arguments `<gpus_per_node> <nodes>` sweep `all_reduce`, `all_gather`, `reduce_scatter` and `alltoall` from 1 KiB to 1 GiB and merge verified `#wrong=0` out-of-place latencies via `tokensim-operator-data import-nccl`.

## 4. Operator timeline collection (`collect_operator_trace.py`)

```bash
CUDA_VISIBLE_DEVICES=0 \
  python -m scripts.collect.collect_operator_trace \
    --model /path/to/local/Qwen3-8B \
    --output outputs/profiler/qwen3-8b \
    --tensor-parallel-size 1 \
    --batch-size 1 \
    --input-tokens 32 \
    --output-tokens 3
```

The collector runs one annotated structure pass to label module ownership, removes all hooks, and captures `--repeats` (default 3) GPU-only timing passes bracketed by unprofiled baselines, pinning SHA-256 digests and selecting the upper-median timing sample in `manifest.json`.
Compare against a simulated trace with `python -m TokenSim.profiler.compare` (see [Operator tracing](../../docs/reference/operator-tracing.md)).

## 5. Validation, merging and calibration

```bash
tokensim-operator-data validate data/operator_data/h100_sxm/vllm
tokensim-operator-data coverage \
  --device h100_sxm \
  --backend vllm \
  --models llama-3-8b \
  --tp 1,2,4,8 \
  --out tmp/coverage-h100.json
tokensim-operator-data calibrate \
  --device h100_sxm \
  --backend vllm \
  --out tmp/calib-h100.yaml
```

To update a subset of tables without replacing an existing package, import into a temporary directory and merge in Python with `merge_packages(OperatorDataPackage.load(target), OperatorDataPackage.load(local), prefer="extra").write(target)`.
