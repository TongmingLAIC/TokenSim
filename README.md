# TokenSim

TokenSim is a discrete-event simulator for large language model (LLM) serving clusters.
It estimates latency, throughput and resource utilization from measured operator costs, analytical models, network topology and key/value (KV) cache behavior.
It runs on a CPU using model configurations and hardware catalogs.

The project paper is [TokenSim: Enabling Hardware and Software Exploration for Large Language Model Inference Systems](https://arxiv.org/abs/2503.08415).

## Install and run

Run these commands from a checkout.
Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
tokensim \
  --latency_backend analytical \
  --request_count 100 \
  --results_path results/example
```

This command simulates 100 synthetic requests for Llama-7B on one A100.
Its default model and cluster are available in a standard virtual environment installed from the wheel.
Guides with relative paths to `data/`, `dataset/`, or `scripts/` use a checkout.
For developer dependencies and checks, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Configure a simulation

| Input | Source |
| --- | --- |
| Model | `--model` accepts a model catalog ID or YAML path. |
| Cluster | `--cluster` selects a JSON file describing workers, topology, parallel layout and storage. |
| Hardware, topology and operator data | The default catalog is in `data/` in a checkout or in the installed data directory. |
| KV transfer | `--kv_transfer_config` selects a connector file as the default for worker groups without their own connector. |
| Workload | Use CLI options or `--workload_config` to specify request arrivals, lengths and targets. |

Use `--data_root` to select another hardware/model catalog.
Set `--cluster` explicitly when using another catalog to select its cluster file.
Run `tokensim --help` to list available arguments.
Read the [documentation index](docs/README.md) for runnable examples of trace workloads, parallel layouts, disaggregated serving, KV storage and operator profiling.

### Workload

Request generation and latency targets are independent of the model.
A workload file can set them as shown below, and explicit CLI values take precedence.
Latency targets are also called service-level objectives (SLOs).

```json
{
  "qps": 10,
  "prefill_mean_len": 256,
  "decode_mean_len": 64,
  "first_token_latency": {
    "max": 10
  },
  "decode_token_latency": {
    "p99": 0.1
  }
}
```

Targets are optional and measured in seconds.
`first_token_latency` applies to the time to first token and `decode_token_latency` to each request's mean inter-token latency.
A `max` target sets a limit for every request.
`p50` sets a limit for the median latency, and `p99` sets a limit for the 99th percentile.

See the [workload guide](docs/guides/workloads.md) for trace inputs, repetition, AgentX warmup and scheduled root arrivals.

## Results

Results are written to `--results_path`, which defaults to `results/<model>/<cluster directory>/<cluster>`.

- `result_<qps>.json`, or `agentx_c<concurrency>.json` for AgentX, summarizes latency, throughput, paused requests, recomputation, prefix reuse, KV transfers, parallel communication and MoE loads.
  It records whether latency estimates come from measurements, interpolation, extrapolation or formulas.
  With latency targets, `slo` reports which targets were met and goodput for requests meeting every `max` target.
- `missing_shapes_<qps>.json` groups operator dimensions that tables could not answer, identifying what to measure in another profiling run.
- `failure.json` records diagnostics for runs stopped before normal completion or with an error, alongside saved partial results.
  `failed.txt` records the error message when the run fails.

## Citation

If you use TokenSim in your research, cite the following paper.

```bibtex
@misc{wu2025tokensimenablinghardwaresoftware,
  title={TokenSim: Enabling Hardware and Software Exploration for Large Language Model Inference Systems},
  author={Feiyang Wu and Zhuohang Bian and Guoyang Duan and Tianle Xu and Junchi Wu and Teng Ma and Yongqiang Yao and Ruihao Gong and Youwei Zhuo},
  year={2025},
  eprint={2503.08415},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2503.08415}
}
```

## License and acknowledgments

TokenSim is licensed under [Apache-2.0](LICENSE).
It builds on SimPy.
Measured kernel tables are imported from NVIDIA AIConfigurator, the AgentX trace parser adapts NVIDIA AIPerf, and the scheduler and KV-cache designs follow vLLM.
Attribution and retained licenses are in [NOTICE](NOTICE), [third_party/licenses](third_party/licenses/) and the [operator-data notices](data/operator_data/LICENSES.md).
