# TokenSim documentation

Start with [installation and quick run](../README.md#install-and-run).
Commands referencing repository files run from a checkout.

## Guides

| Goal | Guide |
| --- | --- |
| Generate synthetic requests or replay trace and agent workloads | [Workloads and AgentX](guides/workloads.md) |
| Place workers and configure parallelism or prefill/decode disaggregation | [Parallelism and worker layouts](guides/serving.md) |
| Reuse prefixes, apply cache policies, transfer KV pages and extend storage backends | [KV caching, transfer and storage](guides/kv-storage.md) |
| Split attention and feed-forward network (FFN) execution across pools | [Attention/FFN disaggregation](guides/attention-ffn.md) |

## Reference

- [Operator latency and data packages](reference/operator-latency.md): table lookup, analytical roofline models, communication models, AIConfigurator imports and result provenance.
- [Models and architectures](reference/models.md): mixture of experts, multi-head latent attention, speculative decoding, Gemma-4 block layouts and Groq LP30 / Rubin modeling.
- [Tracing and profiler comparison](reference/operator-tracing.md): operator timelines, PyTorch Profiler alignment and timeline figures.
- [Device catalog](../data/devices/README.md): supported accelerators, evidence grades and hardware parameters.

## Measurement and examples

- [HBF storage plugin example](../examples/storage/hbf/README.md).
- [Hardware measurement tools](../scripts/collect/README.md).
- [Contributor guide](../CONTRIBUTING.md).
