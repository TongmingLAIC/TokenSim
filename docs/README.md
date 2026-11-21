# TokenSim documentation

Start with the [installation and quick run](../README.md#install-and-run).
Commands that refer to repository files assume a checkout.

## Guides

| Goal | Guide |
| --- | --- |
| Generate requests or replay a trace or agent workload | [Workloads and AgentX](guides/workloads.md) |
| Place workers and configure parallelism or prefill/decode disaggregation | [Parallelism and worker layouts](guides/serving.md) |
| Reuse prefixes and choose programmable KV policies | [Prefix caching](guides/prefix-cache.md) |
| Transfer KV pages or configure memory, SSD and pluggable storage | [KV transfer and storage](guides/kv-storage.md) |
| Split attention and feed-forward (FFN) work across hardware | [Attention/FFN disaggregation](guides/attention-ffn.md) |

## Reference

- [Operator latency model](reference/operator-latency-model.md): measured lookup, analytical estimates, communication and result provenance.
- [Tracing and profiler comparison](reference/operator-tracing.md): timeline recording, collection boundaries and reports.
- [Models](reference/models.md): model layouts, expert routing, latent attention and speculative decoding.
- [LP30 and Rubin](reference/lpx.md): device assumptions, execution and SRAM admission.
- [Operator data packages](reference/operator-data.md): sources, imports, coverage and calibration.
- [Device catalog](../data/devices/README.md): supported devices and hardware inputs.

## Extension and collection

- [Storage backend plugins](development/storage-backends.md) and the [HBF example](../examples/storage/hbf/README.md).
- [Hardware measurement tools](../scripts/collect/README.md).
- [Contributor guide](../CONTRIBUTING.md).
