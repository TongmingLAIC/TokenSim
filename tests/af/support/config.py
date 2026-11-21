"""Small implicit and explicit AF cluster configurations."""

from __future__ import annotations

from TokenSim.config.cluster_config import ClusterConfig


def _cluster_dict(
    *,
    af_block_size=1,
    attention_workers=1,
    ffn_workers=1,
    duplex="full",
    trace_enabled=True,
):
    link = {
        "name": "af-link",
        "attention_group": "a-pool",
        "ffn_group": "f-pool",
        "network": "af-net",
        "resource_group": "fabric",
        "duplex": duplex,
    }
    if duplex == "full":
        link.update({"a_to_f_capacity": 1, "f_to_a_capacity": 1})
    else:
        link["half_duplex_capacity"] = 1
    worker_groups = [
        {
            "name": "a-pool",
            "capabilities": ["af_attention"],
            "hardware": "a-device",
            "num_workers": attention_workers,
            "network": "af-net",
        },
        {
            "name": "f-pool",
            "capabilities": ["af_ffn"],
            "hardware": "f-device",
            "num_workers": ffn_workers,
            "network": "af-net",
        },
    ]
    return {
        "num_workers": sum(group["num_workers"] for group in worker_groups),
        "networks": {"af-net": "profile"},
        "worker_groups": worker_groups,
        "af": {
            "enabled": True,
            "af_block_size": af_block_size,
            "trace_enabled": trace_enabled,
            "communication_domains": {
                "node_size": 8,
                "node_count": (sum(g["num_workers"] for g in worker_groups) + 7) // 8,
                "placement_strategy": "manual",
            },
            "links": [link],
        },
    }


def _load_cluster(**overrides):
    return ClusterConfig(**_cluster_dict(**overrides))


def _replica(group, name, dp_rank, tp_size, ep_size, worker_offset, expert_tp=1):
    return {
        "name": name,
        "rank_mapping": [
            {
                "domain_rank": dp_rank * tp_size + tp_rank,
                "tp_rank": tp_rank,
                "pp_rank": 0,
                "dp_rank": dp_rank,
                "ep_rank": (
                    (dp_rank * tp_size + tp_rank) // expert_tp if ep_size > 1 else 0
                ),
                "physical_worker": {
                    "group_name": group,
                    "worker_index": worker_offset + tp_rank,
                },
            }
            for tp_rank in range(tp_size)
        ],
    }


def _cluster(
    a_tp=2,
    a_dp=2,
    f_tp=2,
    f_dp=2,
    f_ep=None,
    expert_placement_strategy="linear",
    placement_strategy="manual",
):
    f_ep = f_tp * f_dp if f_ep is None else f_ep
    raw = _cluster_dict(attention_workers=a_tp * a_dp, ffn_workers=f_tp * f_dp)
    raw["networks"]["af-net"] = "test"
    af = raw["af"]
    af.pop("trace_enabled")
    af["communication_domains"]["placement_strategy"] = placement_strategy
    if placement_strategy == "dedicated":
        af["communication_domains"]["node_count"] = sum(
            (group["num_workers"] + 7) // 8 for group in raw["worker_groups"]
        )
    af["links"][0].update(
        attention_domain="a-domain",
        ffn_domain="f-domain",
        resource_group="af-fabric",
    )

    domain_layouts = (
        (
            "a",
            a_tp,
            a_dp,
            1,
            (
                ("all_reduce", "a_compute", "a_to_f"),
                ("all_gather", "residual", "f_to_a"),
            ),
        ),
        ("f", f_tp, f_dp, f_ep, (("all_reduce", "expert_compute", "f_to_a"),)),
    )
    domains = []
    for group, (role, tp, dp, ep, collectives) in zip(
        raw["worker_groups"], domain_layouts
    ):
        name = group["name"]
        parallel = dict(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            data_parallel_size=dp,
        )
        if role == "f":
            parallel.update(
                expert_parallel_size=ep,
                expert_placement_strategy=expert_placement_strategy,
            )
        domains.append(
            dict(
                name=f"{role}-domain",
                capability=group["capabilities"][0],
                worker_group=name,
                parallel=parallel,
                logical_instances=[
                    _replica(
                        name, f"{role}{rank}", rank, tp, ep, rank * tp, tp * dp // ep
                    )
                    for rank in range(dp)
                ],
                communication_plan=[
                    dict(
                        operation=operation,
                        after_stage=stage,
                        participants=list(range(tp)),
                        network="af-net",
                        resource_group="af-fabric",
                        resource_lane=lane,
                    )
                    for operation, stage, lane in collectives
                    if tp > 1
                ],
            )
        )
    af["execution_domains"] = domains
    return ClusterConfig(**raw)


def _with_prefill_pool(cluster):
    from dataclasses import replace
    from TokenSim.config.cluster_config import WorkerGroupConfig

    if any("prefill" in group.capabilities for group in cluster.worker_groups):
        return cluster
    first = cluster.worker_groups[0]
    group = WorkerGroupConfig(
        name="prefill-pool",
        role="prefill",
        hardware=first.hardware,
        num_workers=1,
        network=first.network,
    )
    af = replace(
        cluster.af,
        communication_domains=replace(
            cluster.af.communication_domains,
            node_count=cluster.af.communication_domains.node_count + 1,
        ),
    )
    return replace(
        cluster,
        num_workers=cluster.num_workers + 1,
        worker_groups=[*cluster.worker_groups, group],
        af=af,
    )
