"""Measured-key fixture for the absorbed MLA attention core."""


def core_key(**overrides):
    key = dict(
        attn_dtype="bf16",
        kv_cache_dtype="bf16",
        phase="decode",
        execution_mode="absorbed",
        batch_size=1,
        query_len=1,
        context_lengths="[[16384,1]]",
        num_heads=128,
        kv_lora_rank=512,
        rope_head_dim=64,
        nope_head_dim=128,
        v_head_dim=128,
    )
    key.update(overrides)
    return key
