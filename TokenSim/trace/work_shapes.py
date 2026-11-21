"""Useful rank-local work for fused full-model trace segments."""


def attention_components(backend, requests):
    model = backend.model
    components = []
    for request in requests:
        query = request.query_tokens
        context = request.num_computed_tokens + query
        key = dict(
            batch_size=1,
            num_heads=backend.q_dim_local // model.head_dim,
            num_kv_heads=backend.kv_dim_local // model.head_dim,
            head_dim=model.head_dim,
            attn_dtype=model.activation_dtype,
            kv_cache_dtype=model.kv_cache_dtype,
            window_size=model.sliding_window,
        )
        if request.is_context_build:
            table = "context_attention"
            key.update(input_seq_len=query, context_len=context)
        else:
            table = "generation_attention"
            key["context_len"] = context
        components.append(dict(name="attention", table=table, key=key))
    return components


def moe_key(backend, tokens):
    model = backend.model
    return dict(
        dtype=model.operator_dtype("routed_experts"),
        hidden_size=model.hidden_size,
        inter_size=model.moe_intermediate_size,
        top_k=model.moe.num_experts_per_tok,
        num_experts=model.moe.num_experts,
        tp_size=backend.moe_tp,
        ep_size=backend.moe_ep,
        num_tokens=max(1, int(round(tokens * backend.moe_token_multiplier))),
    )
