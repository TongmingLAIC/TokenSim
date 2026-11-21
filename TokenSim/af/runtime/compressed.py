"""Native target-only compressed model ownership and grouped KV admission."""

from collections import defaultdict
from dataclasses import replace
from math import ceil

from TokenSim.af.runtime.records import AFKVHeadPartition, AFModelComponent
from TokenSim.block.hybrid.kv_layout import hybrid_kv_layout
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.config.weight_memory import packed_weight_memory
from TokenSim.errors import ConfigurationError


def compressed_layout(model, block_size, tokens):
    if model.components is None:
        return None
    if model.dspark is not None:
        raise ConfigurationError(
            "compressed AF requires a target-only model; DSpark split verification is unsupported"
        )
    return hybrid_kv_layout(
        model,
        block_size=block_size,
        max_num_batched_tokens=tokens,
        max_concurrent_batches=1,
        speculative_tokens=0,
        include_mtp=False,
    )


def compressed_partitions(layout, tp):
    # Latent KV, indexer keys, compressor states and SWA are replicated under TP.
    return tuple(
        AFKVHeadPartition(
            tp_rank=rank,
            global_head_ids=(0,),
            replica_factor=tp,
            bytes_per_token=layout.physical_slot_bytes // layout.logical_block_tokens,
            bytes_per_block=layout.physical_slot_bytes,
        )
        for rank in range(tp)
    )


def layer_residence(layout, context):
    """Charge native padded physical slots, including every bounded state group."""
    result = defaultdict(int)
    for group in layout.groups:
        slots = group.max_request_slots(context, layout.max_in_flight_tokens)
        padding = layout.physical_slot_bytes - group.group_bytes
        for index, page in enumerate(group.layers):
            layer = int(page.name.split(".")[1])
            result[layer] += slots * (page.page_bytes + (padding if index == 0 else 0))
    return dict(result)


def components_for_rank(model, domain, mapping, expert_ids, cache, router_location):
    """Partition the existing exact packed inventory; never infer V4 GQA weights."""
    tp = domain.parallel.tensor_parallel_size
    if domain.parallel.pipeline_parallel_size != 1:
        raise ConfigurationError("compressed AF currently requires PP1")

    def report(size):
        if size not in cache:
            cache[size] = packed_weight_memory(
                model, ParallelConfig(tensor_parallel_size=size), include_mtp=False
            )
        return cache[size]

    a_side = domain.capability == "af_attention"
    grouped = defaultdict(list)
    tensors = report(tp)["tensors"]
    for tensor in tensors:
        category = tensor["component"]
        if category == "routed_expert":
            continue
        router = category in {"router", "hash_routing"}
        owner_a = category not in {"shared_expert", "router", "hash_routing"}
        if router:
            owner_a = router_location == "a"
            if mapping.tp_rank != 0:
                continue
        if owner_a != a_side:
            continue
        name = tensor["name"]
        layer = int(name.split(".")[1]) if name.startswith("layers.") else None
        # Existing execution validation expects one shared component per layer.
        category = "router" if router else category
        grouped[(layer, category)].append(tensor)
    components = []
    for (layer, category), values in grouped.items():
        payload = sum(t["weight_bytes"] + t["metadata_bytes"] for t in values)
        scales = sum(t["scale_bytes"] for t in values)
        shape = (("hidden_size", model.hidden_size), ("tp_size", tp))
        components.append(
            AFModelComponent(
                component_id=f"compressed.{layer}.{category}.tp{mapping.tp_rank}-of-{tp}",
                category=category,
                layer_id=layer,
                bytes=payload + scales,
                scale_bytes=scales,
                formula="sum native packed tensors, metadata and scales; target only",
                formula_source="TokenSim.config.weight_memory.packed_weight_memory",
                shape=shape,
                storage_dtype="native_mixed",
            )
        )
    if not a_side:
        etp = domain.expert_parallel.expert_tensor_parallel_size
        values = defaultdict(list)
        for tensor in report(etp)["tensors"]:
            if tensor["component"] == "routed_expert":
                values[int(tensor["name"].split(".")[1])].append(tensor)
        for layer, tensors in values.items():
            total = sum(t["weight_bytes"] + t["scale_bytes"] for t in tensors)
            scales = sum(t["scale_bytes"] for t in tensors)
            if total % model.moe.num_experts or scales % model.moe.num_experts:
                raise ConfigurationError(
                    "packed expert inventory cannot be partitioned exactly"
                )
            for expert in expert_ids:
                components.append(
                    AFModelComponent(
                        component_id=f"compressed.{layer}.routed_expert.{expert}.tp{mapping.tp_rank % etp}-of-{etp}",
                        category="routed_expert",
                        layer_id=layer,
                        bytes=total // model.moe.num_experts,
                        scale_bytes=scales // model.moe.num_experts,
                        storage_dtype=model.operator_dtype("routed_expert"),
                        formula="native packed expert tensor shard with scale metadata",
                        shape=(
                            ("expert_id", expert),
                            ("hidden_size", model.hidden_size),
                        ),
                    )
                )
    return tuple(components)


def runtime_buffers(model, roles, *, tokens, f_tokens, tp):
    """Named conservative role workspaces; persistent weights and KV are separate."""
    spec, h = model.components, model.hidden_size
    result = []
    if "af_attention" in roles:
        # Same persistent gather/radix and RoPE geometry as the T1 memory model.
        index_bytes = spec.index_head_dim // 2 + ceil(spec.index_head_dim / 32)
        result.extend(
            (
                (
                    "compressed_rope",
                    model.max_position_embeddings * spec.qk_rope_head_dim * 4,
                ),
                (
                    "indexer_gather_and_radix",
                    model.max_position_embeddings * 10 * index_bytes + 2**20,
                ),
                ("flashinfer_workspace", 128 * 2**20),
                ("index_topk", tokens * spec.index_topk * 4),
                ("hyper_residual_double_buffer", 2 * tokens * spec.hc_mult * h * 2),
                ("attention_query_staging", tokens * model.q_dim // tp * 5),
                (
                    "compressed_projection_staging",
                    tokens
                    * (
                        spec.q_lora_rank
                        + model.head_dim
                        + spec.o_groups * spec.o_lora_rank // tp
                    )
                    * 2,
                ),
            )
        )
    if "af_ffn" in roles:
        result.extend(
            (
                (
                    "expert_gate_up_staging",
                    f_tokens
                    * model.moe.num_experts_per_tok
                    * model.moe_intermediate_size
                    * 4,
                ),
                (
                    "expert_dispatch_staging",
                    f_tokens * model.moe.num_experts_per_tok * (h * 2 + 12),
                ),
            )
        )
    return tuple(result)
