"""A-side compressed modules, hyper-residual sites and exact precision work."""

from collections import Counter, defaultdict
from types import SimpleNamespace

from TokenSim.af.providers.compute import OperatorComponent, _worst_match
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.errors import ConfigurationError
from TokenSim.latency.attention.estimator import CompressedAttentionEstimator
from TokenSim.latency.attention_work import CompressedAttentionWork
from TokenSim.operator_data.lookup import LookupPolicy, OperatorLookup


def _state(provider, resolver, tp):
    if not hasattr(provider, "_compressed_states"):
        provider._compressed_states = {}
    identity = (resolver.device.device_id, tp)
    if identity not in provider._compressed_states:
        model = provider.model
        backend = SimpleNamespace(
            model=model,
            device=resolver.device,
            package=resolver.package,
            parallel_config=ParallelConfig(tensor_parallel_size=tp),
            stats=resolver.stats,
        )
        estimator = CompressedAttentionEstimator(backend) if resolver.package else None
        provider._compressed_states[identity] = (
            estimator,
            CompressedAttentionWork(model, resolver.device),
        )
    return provider._compressed_states[identity]


def _component(
    name,
    table,
    key,
    latency,
    resolver,
    work,
    bounds,
    *,
    match="analytical",
    source=None,
):
    return OperatorComponent(
        name=name,
        table=table,
        key=key,
        latency_us=latency,
        match_type=match,
        source_id=source or f"analytical:{resolver.device.device_id}:compressed",
        dataset_version=resolver.dataset_version
        or f"analytical:{resolver.device.device_id}",
        work=work,
        analytical_bounds=bounds,
    )


def _attention(provider, resolver, query, tokens, tp):
    model = provider.model
    layer = model.layer_specs(include_mtp=False)[query.layer_id]
    estimator, sol = _state(provider, resolver, tp)
    lengths = tuple(query.context_lengths)
    queries = tuple(query.query_lengths) or (1,) * len(lengths)
    if (
        len(lengths) != len(queries)
        or sum(queries) != tokens
        or any(
            type(q) is not int or type(k) is not int or q < 1 or k < q
            for q, k in zip(queries, lengths)
        )
    ):
        raise ConfigurationError(
            "compressed attention requires one bounded KV/query pair per request"
        )
    context_groups, decode = Counter(), Counter()
    prefill = query.phase == "prefill" or any(q > 1 for q in queries)
    for q, k in zip(queries, lengths):
        if prefill:
            context_groups[(q, k - q)] += 1
        else:
            decode[(k, q)] += 1
    groups = Counter(zip(queries, lengths))
    work_by_dtype = defaultdict(float)
    compute_us = memory_us = analytic_latency = 0.0
    prefill_parts = []
    shared_weight_bytes = 0.0
    configs = {
        "projection": sol.config.projection_dtype,
        "output_absorption": sol.config.output_absorption_dtype,
        "compressor": sol.config.compressor_dtype,
        "indexer": sol.config.indexer_dtype,
        "indexer_weights": "bf16",
        "attention": sol.config.attention_dtype,
    }
    for (q, k), count in groups.items():
        key = dict(
            phase="context" if prefill else "generation",
            batch_size=count,
            input_seq_len=q,
            context_len=k - q if prefill else k,
            compress_ratio=layer.compress_ratio,
            num_heads=model.num_attention_heads // tp,
            tp_size=tp,
        )
        work = sol.work(key)
        for name, flops in work.flops.items():
            work_by_dtype[configs[name]] += flops
        compute_us += work.math_seconds * 1e6
        memory_us += work.memory_seconds * 1e6
        # Decode prices one full query tensor, with a length-weighted KV term.
        full = sol.work(dict(key, batch_size=tokens)) if not prefill else work
        analytic_latency += (count / tokens if not prefill else 1) * full.latency_us
        if prefill:
            common, phase = sol.shared_phase_us(key)
            prefill_parts.append((q * count, work.latency_us / (common + phase), phase))
            shared_weight_bytes = sum(
                value for name, value in work.bytes.items() if "weights" in name
            )
    if prefill and len(groups) > 1:
        memory_us -= (len(groups) - 1) * shared_weight_bytes / sol.bandwidth * 1e6
        common_key = dict(
            key, phase="generation", batch_size=tokens, input_seq_len=1, context_len=0
        )
        shared = sol.shared_phase_us(common_key)[0]
        efficiency = sum(rows * eta for rows, eta, _ in prefill_parts) / tokens
        composed = shared * efficiency + sum(
            eta * phase for _, eta, phase in prefill_parts
        )
        analytic_latency = max(composed, compute_us, memory_us)
    source = None
    match = "analytical"
    latency = analytic_latency + float(
        resolver.device.analytical.get("kernel_launch_us", 0)
    )
    if estimator:
        estimator.trace_layer_costs = {}
        estimator.estimate_batch(context_groups, decode)
        latency = estimator.trace_layer_costs[layer.attention_kind]
        matches = []
        sources = []
        for (q, k), count in groups.items():
            sample = estimator._query(
                layer.attention_kind,
                "context" if prefill else "generation",
                count if prefill else tokens,
                q,
                k - q if prefill else k,
                record_stats=False,
            )
            matches.append(sample.match)
            sources.append(sample.source_id)
        match = _worst_match(matches)
        source = "compose:" + ":".join(dict.fromkeys(sources))
    key = dict(
        layer_kind=layer.attention_kind,
        compress_ratio=layer.compress_ratio,
        tp_size=tp,
        query_lengths=queries,
        context_lengths=lengths,
        logical_query_rows=tokens,
        forward_phase=query.phase,
        module_scope="projections+compressor+indexer+sparse_attention+output",
    )
    return _component(
        "compressed_attention",
        "dsv4_attention",
        key,
        latency,
        resolver,
        work_by_dtype,
        dict(compute_us=compute_us, memory_us=memory_us),
        match=match,
        source=source,
    )


def _mhc(provider, resolver, tokens, operation, *, hidden=None):
    model, spec, device = provider.model, provider.model.components, resolver.device
    h, hc = hidden or model.hidden_size, spec.hc_mult
    sites = 1
    # The source pre/post tables aggregate two sites. Each explicit site owns half.
    flops = (
        tokens
        * (
            2 * hc * (hc + 2) * hc * h
            + 8 * hc * h
            + spec.hc_sinkhorn_iters * hc * hc * 8
        )
        if operation == "pre"
        else tokens * h * hc * (2 * hc + 2)
    )
    traffic = tokens * h * (2 * hc + 2) * 2
    if operation == "pre":
        traffic += hc * (hc + 2) * hc * h * 4
    compute = flops / device.peak_compute_for("fp32").value * 1e6
    memory = traffic / device.memory_bandwidth_bytes_per_s.value * 1e6
    key = dict(
        op_name=operation, hidden_size=h, hc_mult=hc, num_tokens=tokens, num_sites=sites
    )
    latency, match, source = (
        max(compute, memory) + float(device.analytical.get("kernel_launch_us", 0)),
        "analytical",
        None,
    )
    binding = model.operator_modules.get("mhc")
    if resolver.package and binding:
        native_key = {
            "op_name": operation,
            **binding["key"],
            "kernel_source": binding["kernels"][operation],
            "hidden_size": model.hidden_size,
            "hc_mult": hc,
            "sinkhorn_iters": spec.hc_sinkhorn_iters,
            "num_tokens": tokens,
        }
        cache = getattr(provider, "_mhc_lookups", None)
        if cache is None:
            cache = provider._mhc_lookups = {}
        lookup = cache.get(id(resolver.package))
        if lookup is None:
            lookup = cache[id(resolver.package)] = OperatorLookup(
                resolver.package,
                LookupPolicy(extrapolate="analytical"),
                analytical_scaler=lambda _t, k: float(k["num_tokens"]),
            )
        row = lookup.lookup(binding["table"], native_key)
        latency = row.latency_us / binding["key"]["num_sites"] * h / model.hidden_size
        match, source = row.match_type, row.source_id
        resolver.stats.record(binding["table"], match)
        key = dict(
            native_key,
            charged_sites=sites,
            source_site_divisor=binding["key"]["num_sites"],
            charged_hidden_size=h,
        )
    return _component(
        f"mhc_{operation}_site",
        "dsv4_mhc",
        key,
        latency,
        resolver,
        {"fp32": float(flops)},
        dict(compute_us=compute, memory_us=memory),
        match=match,
        source=source,
    )


def compressed_components(provider, resolver, query, tokens, tp, dtype):
    model = provider.model
    if model.components is None:
        return None
    if query.stage in {"a", "attention"}:
        return [
            _mhc(provider, resolver, tokens, "pre"),
            provider._elementwise(
                resolver,
                "attention_norm",
                "rmsnorm",
                tokens,
                model.hidden_size,
                model.operator_dtype("norm"),
            ),
            _attention(provider, resolver, query, tokens, tp),
        ]
    if query.stage == "layout" and query.operation == "post_attention_norm":
        return [
            _mhc(provider, resolver, tokens, "post"),
            _mhc(provider, resolver, tokens, "pre"),
            provider._elementwise(
                resolver,
                "ffn_norm",
                "rmsnorm",
                tokens,
                model.hidden_size,
                model.operator_dtype("norm"),
            ),
        ]
    if query.stage == "residual":
        return [
            _mhc(provider, resolver, tokens, "post", hidden=model.hidden_size // tp)
        ]
    if (
        query.stage == "router"
        and model.layer_specs(include_mtp=False)[query.layer_id].moe_routing == "hash"
    ):
        return [
            provider._elementwise(
                resolver,
                "hash_routing_lookup",
                "embedding_lookup",
                tokens,
                model.moe.num_experts_per_tok,
                "bf16",
            )
        ]
    if query.stage == "layout" and query.operation == "lm_head":
        return [
            provider._gemm(
                resolver,
                "lm_head",
                tokens,
                model.vocab_size // tp,
                model.hidden_size,
                model.operator_dtype("lm_head"),
            )
        ]
    if query.stage == "layout" and query.operation == "final_norm":
        from TokenSim.latency.output import ResidualOutputEstimator

        backend = SimpleNamespace(
            model=model, device=resolver.device, stats=resolver.stats
        )
        head = ResidualOutputEstimator(backend, scope="target")
        h, hc = model.hidden_size, model.components.hc_mult
        flops = tokens * h * (2 * hc * hc + 8 * hc)
        traffic = head.weight_bytes + tokens * h * (2 * hc + 2)
        reduction = _component(
            "hc_head",
            "target_hc_head",
            dict(num_tokens=tokens, hidden_size=h, hc_mult=hc),
            head.estimate(tokens),
            resolver,
            {"fp32": float(flops)},
            dict(
                compute_us=flops / resolver.device.peak_compute_for("fp32").value * 1e6,
                memory_us=traffic
                / resolver.device.memory_bandwidth_bytes_per_s.value
                * 1e6,
            ),
        )
        return [
            reduction,
            provider._elementwise(
                resolver,
                "final_norm",
                "rmsnorm",
                tokens,
                h,
                model.operator_dtype("norm"),
            ),
        ]
    return None
