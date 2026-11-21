# DeepSeek-V4 attention/FFN execution

The compressed-attention AF path executes the complete DeepSeek-V4-Pro target stack with ordinary autoregressive decoding.
It supports the model's compressed and sliding attention, indexer, compressor state, hyper-residual mixing, hash-routing layers, learned routing, shared experts and routed experts.
The regular engine also derives the default AF activation precision from `activation_dtype` when its checkpoint weights use `fp8_block`.
An enabled DSpark model is rejected because a split draft/verification contract is required before speculative progress can be credited.

## Execution ownership

The A role owns attention projections, CSA/HCA attention, the indexer, compressors, normalization, hyper-residual parameters and state, embedding, final reduction and vocabulary output.
The F role owns shared and routed expert tensors and executes concrete expert-assignment histograms.
The router follows `af.router_location`.
Hash-routing layers perform a lookup and retain the native mapping-table storage.
The synthetic choice excludes batch and scheduling identities, so matched placements receive the same request/token/layer assignments.

The compiler partitions `packed_weight_memory`'s native tensor report, including row packing, block scales, metadata, replicated projections and TP/EP shards.
Shared experts retain their configured TP ownership and EP replication.
Weight admission runs before request or performance selection.
Every physical rank deducts its own component tensors and named protocol/activation buffers from its usable device memory.
Mixed inventories retain one physical device identity per worker.

## Grouped KV and activation admission

Compressed AF uses the existing `hybrid_kv_layout` with its native full, sliding, indexer and compressor-state groups.
Latent KV and auxiliary state are replicated across A TP ranks.
All groups share the padded physical slot stride.
Resident admission charges the exact number of bounded and full group slots for the requested peak context on every required physical rank.
The aggregate byte-per-token field is a reporting interface; group slots determine admission.
Compressed AF reports zero unconfigured host-swap slots.
The engine's grouped replica manager mirrors a logical reservation across the A TP ranks using their minimum admissible pool.

`af.max_inflight_blocks` bounds concurrently active microbatches per A-owner iteration.
Zero preserves unrestricted behavior for ordinary AF models.
Compressed AF requires a positive value so its residual and activation reservations have a finite bound.
The study uses three blocks as a candidate and sweeps other positive values.
The physical memory ledger includes all live A blocks and an F dispatch envelope derived from the active A owners and F batch limit.
This simulated pipeline setting is independent of host process concurrency.

## Latency modules and communication

Measured attention uses the existing complete compressed-attention estimator.
Its module already includes projections, compression, indexing, sparse attention and output projection.
The AF composition excludes additional generic QKV or dense-attention work.
The same compressed FLOP/byte geometry provides analytical estimates when that mode is selected.
Ragged analytical prefill shares common projection traffic once and retains the complete logical FLOPs and physical compute/memory lower bounds.
Borrowed operator packages retain their original generation metadata and measured scope.

Measured mHC pre/post tables cover two layer sites.
The AF path allocates half of each two-site estimate to each explicit site and preserves the original total without adding overlap credit.
The final residual post operation processes the rank-local hidden shard.
The subsequent collective transfers all `hc_mult` residual channels.
FFN activation transfers carry the ordinary hidden vector.
The final hyper-residual output reduction uses the existing analytical head estimator.
Per-site timing allocation and TP-local post scaling are explicit composition assumptions.

Dynamic F batches use their exact routed histogram and packed weight traffic.
A balanced whole-layer MoE table does not identify the concrete merged histogram, so compressed AF uses the analytical `routed_experts` contract for this stage.
Native attention/mHC tables and eligible ordinary GEMM rows remain usable independently.
Non-compressed AF models keep their original lookup behavior.

## Serving and configuration

Use the `deepseek-v4-pro` catalog model with an attention/FFN cluster.
See [attention/FFN disaggregation](afd.md) for worker groups, links and routing configuration.
`AFService` executes chunked prefill and ordinary autoregressive decode through the same target stack.
Live prefixes remain resident while callers pause a request.
`max_inflight_blocks` controls how many microbatches an owner may have in flight.
See [service variability](af-jitter.md) for timing, receive buffers and credit scope.

The `deepseek-v4-pro-0813-dspark` catalog entry models a three-layer DSpark draft on ordinary serving workers.
Draft memory, verification latency and lookahead capacity use the declared draft configuration.
Split attention/FFN verification does not support DSpark yet.
