"""Physical inventory and expert observations for independent serving domains."""

from types import SimpleNamespace


def domain_workers(engine):
    groups = {}
    for worker in getattr(engine, "workers", []):
        domain = getattr(worker, "execution_domain", None)
        if domain is not None:
            groups.setdefault(domain, []).append(worker)
    return groups


def parallel_domain_stats(engine):
    groups = domain_workers(engine)
    if not groups:
        return {}
    domains = {}
    for name, workers in groups.items():
        ids = {w.id for w in workers}
        domains[name] = dict(
            parallel_config=workers[0].parallel_config.to_dict(),
            expected_ranks=workers[0].parallel_config.world_size,
            worker_ids=sorted(ids),
            groups=workers[0].parallel_communicator.describe(),
            settings={
                key: getattr(workers[0].config, key)
                for key in [
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "gpu_memory_utilization",
                    "long_prefill_token_threshold",
                    "prefill_schedule_interval",
                    "mtp_num_speculative_tokens",
                    "mtp_acceptance_length",
                ]
            },
            placements={
                role: pool.placements_in_domain(name)
                for role, pool in [
                    ("prefill", engine.prefill_workers),
                    ("decode", engine.decode_workers),
                ]
                if any(w.id in ids for w in pool.workers)
            },
        )
    return dict(
        parallel_config=dict(
            mode="independent_serving_domains",
            domains={name: value["parallel_config"] for name, value in domains.items()},
        ),
        parallel_expected_rank_count=sum(
            value["expected_ranks"] for value in domains.values()
        ),
        parallel_serving_domains=domains,
    )


def moe_domain_stats(engine, builder):
    groups = domain_workers(engine)
    if not groups:
        return {}
    reports = {}
    for name, workers in groups.items():
        view = SimpleNamespace(
            workers=workers,
            expert_placement=workers[0].expert_placement,
            moe_config=engine.moe_config,
        )
        reports[name] = builder(view, split_domains=False)
    return dict(
        moe_ep_rank_count=sum(r["moe_ep_rank_count"] for r in reports.values()),
        moe_expert_placement=dict(
            domains={
                name: report["moe_expert_placement"] for name, report in reports.items()
            }
        ),
        moe_per_rank_expert_load={
            f"{name}:{rank}": count
            for name, report in reports.items()
            for rank, count in report["moe_per_rank_expert_load"].items()
        },
        moe_serving_domains=reports,
    )
