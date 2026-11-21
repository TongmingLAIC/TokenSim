# Independent complete-model serving domains

Dedicated prefill and decode worker groups can each declare `parallel_config` and `kv_transfer` in the cluster configuration.
The global parallel layout stays omitted when group layouts are present.
Every group has its own TP, DP and expert placement, and physical worker identifiers remain unique across the entire allocation.
Collective discovery never crosses a complete-model group boundary.
P/D handoff selects decode replicas in the destination group's DP coordinate space.
The source coordinate remains attached during DMA, and delivery assigns the selected decode coordinate.
Transport lifecycle rows retain both source and destination DP ranks for independent groups.
DP lockstep maintains a separate busy wave and finished-state consensus for each group.
The feature is independent of AF execution domains and cannot be combined with AF in the same configuration.

```json
{
  "num_workers": 16,
  "topology": "gb200_nvl72",
  "networks": {"fabric": "ethernet800Gb"},
  "worker_groups": [
    {
      "name": "P",
      "role": "prefill",
      "hardware": "gb200",
      "num_workers": 8,
      "network": "fabric",
      "parallel_config": {
        "data_parallel_size": 8,
        "data_parallel_size_local": 8,
        "enable_expert_parallel": true,
        "expert_parallel_scope": "global",
        "expert_parallel_size": 8
      }
    },
    {
      "name": "D",
      "role": "decode",
      "hardware": "gb200",
      "num_workers": 8,
      "network": "fabric",
      "parallel_config": {
        "data_parallel_size": 8,
        "data_parallel_size_local": 8,
        "enable_expert_parallel": true,
        "expert_parallel_scope": "global",
        "expert_parallel_size": 8
      }
    }
  ],
  "kv_transfer": {
    "kv_connector": "NoopConnector",
    "kv_connector_extra_config": {
      "pd_endpoint_limits": {
        "send_bytes_per_second": 50000000000,
        "receive_bytes_per_second": 50000000000,
        "fabric_bytes_per_second": 400000000000
      }
    }
  }
}
```

The example uses declared analytical transport limits and makes no hardware accuracy claim.
Ordinary workers may use independent TP layouts for complete-model execution.
Endpoint transport currently supports TP1 ordinary workers, including independent DP-attention and global EP groups.
The grouped transport requires identical native KV geometry at both ends and PP1.
It copies the resident computed group pages using their actual physical slot stride, including padding.
It does not charge a uniform full-context page for each layer.

Source DMA references survive source request release.
The receiver allocates private request-owned pages before DMA starts, and DMA holds separate references until physical completion.
Receiver capacity can delay the transfer without publishing a destination prefix.
Copy completion publishes only cache identities already proven complete at the source.
Partial prompt tails remain private and are computed further by the normal decode scheduler.
Cancellation drains the active transfer, releases both sets of references and publishes no cancelled destination data.
The native scheduler resumes imported pages through its loaded-request queue without repeating prefill.

The request's first token is sampled when prefill completes.
Transfer and receiver scheduling occur after that sample and contribute to the following inter-token interval and end-to-end latency.
This timing convention must be checked against a reference frontend's streaming behavior before an accuracy comparison.

Results include `parallel_serving_domains`, `moe_serving_domains` and `pd_transport`.
Physical capacity reporting distinguishes identical DP coordinates in different domains.
Per-domain reports preserve each layout, expert ownership, rank loads and placement counts.
The transport report retains physical endpoint flows, bandwidth-sharing intervals, lifecycle events and outstanding ownership counts.
