# HBF storage backend example

This example package (`tokensim_hbf:HBFBackend`) models high-bandwidth flash (HBF) as an external KV offload tier using media parameters from the [HBF characterization study](https://arxiv.org/abs/2608.11668).

## Install and run

```bash
python -m pip install -e examples/storage/hbf
tokensim \
  --model data/models/llama-7b.yaml \
  --cluster data/clusters/8_h100/tp8.json \
  --latency_backend analytical \
  --batching paged-attn \
  --block_size 16 \
  --workload_type qwen_jsonl \
  --dataset_path dataset/qwen_example.jsonl \
  --request_count 3 \
  --kv_transfer_config data/kv_transfer/mooncake_store_hbf.json \
  --verbose none \
  --results_path results/hbf-example
```

Use `MooncakeStoreConnector` for step-blocking uniform I/O or `UniformAsyncStore` for queued restoration and background writes (see [KV caching, transfer and storage](../../../docs/guides/kv-storage.md#storage-backend-plugins)).

## Media configuration

```json
{
  "offload": {
    "backend": "tokensim_hbf:HBFBackend",
    "name": "hbf",
    "config": {
      "capacity_bytes": 1099511627776,
      "seq_read_bandwidth_bytes_per_s": 137438953472,
      "seq_write_bandwidth_bytes_per_s": 137438953472,
      "read_latency_us": 12,
      "write_latency_us": 120
    }
  }
}
```

Optional `random_4k_read_bandwidth_bytes_per_s` and `random_4k_write_bandwidth_bytes_per_s` default to 80% of sequential bandwidth.
For `auto` access patterns, effective bandwidth interpolates logarithmically between 4 KiB and 10 MiB transfer sizes.
Results report counters under `storage_metrics[master]["hbf"]`.
For HBF GPU main-memory variants, see [Device catalog](../../../data/devices/README.md#high-bandwidth-flash-hbf-gpu-memory-variants).
