"""Reusable LP30 live scratch reservations independent of weights and KV leases."""

from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device.lpx import LP30Memory


def workspace_components(components, model, policy):
    if not isinstance(policy, LP30Memory):
        policy = LP30Memory(**policy)
    tokens = policy.prefill_chunk
    activation_bytes = dtype_size_bytes(model.activation_dtype, role="activation_dtype")
    cache_bytes = dtype_size_bytes(model.kv_cache_dtype, role="kv_cache_dtype")
    attention, ffn, boundary, auxiliary = 0, 0, 0, 0
    for component in components:
        shape = dict(component.shape)
        if component.category == "attention":
            layout = model.block_layout
            if layout.to_dict():
                auxiliary += 2 * model.hidden_size * activation_bytes
                auxiliary += layout.parameter_elements(model.hidden_size) * max(
                    0,
                    activation_bytes
                    - dtype_size_bytes(model.dtype, role="activation_dtype"),
                )
                auxiliary += (
                    2
                    * shape["head_dim"]
                    * int(model.qk_norm)
                    * max(
                        0,
                        activation_bytes
                        - dtype_size_bytes(model.dtype, role="activation_dtype"),
                    )
                )
            q = shape["q_heads_local"] * shape["head_dim"]
            kv = shape["kv_heads_local"] * shape["head_dim"]
            tile = min(policy.maximum_context, policy.attention_tile_keys)
            qkv = tokens * (q + 2 * kv) * activation_bytes
            # QK scores, online max/sum/output state and double-buffered KV tiles.
            scores = tokens * shape["q_heads_local"] * tile * 4
            online = tokens * (q + 2 * shape["q_heads_local"]) * 4
            cache_tile = (
                policy.tile_buffers * tile * 2 * kv * (cache_bytes + activation_bytes)
            )
            attention = max(attention, qkv + scores + online + cache_tile)
        elif component.category == "dense_ffn":
            hidden, width = shape["hidden_size"], shape["local_intermediate_size"]
            # Gate/up and activation outputs coexist with down-projection inputs.
            ffn = max(ffn, tokens * (3 * width + 2 * hidden) * activation_bytes)
        elif component.category == "embedding_lm_head":
            if model.block_layout.embedding_scale:
                auxiliary += model.hidden_size * activation_bytes
            logits = (model.vocab_size + shape["tp_size"] - 1) // shape["tp_size"]
            boundary = max(boundary, tokens * logits * (activation_bytes + 4))
    # Layers and attention/FFN execute sequentially; their scratch arena is reused.
    return (
        ("lp30_live_scratch", max(attention, ffn, boundary)),
        ("lp30_unallocated_reserve", policy.reserve_bytes),
        ("lp30_auxiliary_weights", auxiliary),
    )


def memory_policies(hardware, workers):
    return tuple(
        policy
        for name in sorted(
            {
                worker.hardware
                for worker in workers
                if {"af_attention", "af_ffn"}.intersection(worker.capabilities)
            }
        )
        if (policy := hardware.device(name).lpx_memory) is not None
    )


def validate_queries(policies, query_tokens, context):
    for policy in policies:
        if query_tokens > policy.prefill_chunk or context > policy.maximum_context:
            raise ConfigurationError(
                "LP30 query exceeds its reserved prefill chunk or maximum context"
            )
