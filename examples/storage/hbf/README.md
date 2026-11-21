# Optional HBF storage backend

This example models high-bandwidth flash (HBF) as external KV storage behind the TokenSim plugin interface.
The selected cluster keeps its normal accelerator configuration.
The media assumptions come from the [HBF characterization study](https://arxiv.org/abs/2608.11668).

## Install and run

Run these commands from the TokenSim checkout after installing TokenSim.

```bash
python -m pip install -e examples/storage/hbf
```

Use a normal cluster configuration and select HBF through `--kv_transfer_config`.
This example uses eight standard H100 GPUs, a 1 TiB HBF store and an assumed sequential bandwidth of 128 GiB/s in each direction.

```bash
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

The example selects analytical latency; standard GPUs can also use their measured operator tables.
The trace supplies prefix hashes for store writes and reuse.
Reads from HBF occur when a reusable prefix has been evicted from the GPU cache.

The example uses the blocking uniform mode of `MooncakeStoreConnector`.
Select `UniformAsyncStore` to use queued restoration and background writes with the same plugin.
See [KV transfer and storage](../../../docs/guides/kv-storage.md) for connector requirements.

## Media configuration

The connector's `offload` object selects `tokensim_hbf:HBFBackend` and supplies plugin-owned options in `config`.
The example configurations name the instance `hbf`.

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

Capacity uses `capacity_bytes`.
Bandwidth fields use bytes per second: `seq_read_bandwidth_bytes_per_s`, `seq_write_bandwidth_bytes_per_s`, `random_4k_read_bandwidth_bytes_per_s` and `random_4k_write_bandwidth_bytes_per_s`.
Fixed media latency uses `read_latency_us` and `write_latency_us`.
The plugin validates these fields and supplies their defaults.
Each operation costs its fixed latency plus transferred bytes divided by effective bandwidth.
Missing random 4 KiB bandwidth values default to 80% of sequential bandwidth.
The effective bandwidth interpolates logarithmically between 4 KiB and 10 MiB operation sizes and remains at the endpoint values outside that interval.

The connector provides cache admission, readiness, eviction and pinning.
Results report media counters under `storage_metrics[master]["hbf"]` and the configured media profile under `mooncake_offload_profiles`.
See the [plugin contract](../../../docs/development/storage-backends.md) to implement a different backend.
The [device catalog](../../../data/devices/README.md#hbf-memory-variants) lists independent GPU memory variants.
