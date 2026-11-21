"""Shared asynchronous admission over grouped or fully allocated uniform pages."""

from TokenSim.llm.llm_request import RequestStatus


def has_handoff_prefix(req):
    return (
        req.status is RequestStatus.WAITING_FOR_KV
        and not req.is_prefill
        and not req.needs_recompute
    )


def admit_async(scheduler, req, local_plan, external):
    transferred = req.num_computed_tokens if has_handoff_prefix(req) else None
    manager = scheduler.block_manager
    manager.allocate(req)
    local = local_plan.hit_tokens
    if not manager.grouped_cache:
        local = min(local, max(0, req.context_len - 1))
    req.gpu_cached_context_tokens = local
    req.gpu_cached_prefill_tokens = min(req.prefill_len, local)
    req.external_cached_prefill_tokens = 0
    req.external_cached_context_tokens = 0
    if external and external[1]:
        scheduler.connector.begin_load(req, *external)
        req.record_context_admission(local, req.external_cached_context_tokens)
        scheduler.inflight_prefills[req.id] = req
        return
    if req.needs_recompute:
        req.recompute_tokens = max(0, req.context_len - local)
    req.start_context_build(local if transferred is None else transferred)
    req.record_context_admission(local, 0)
    req.status = RequestStatus.RUNNING


def resume_uniform(scheduler, req, tokens, query_tokens, step):
    # Uniform admission already owns every page of this request's context.
    scheduler.loaded_waiting.pop(0)
    scheduler.loading.pop(req.id, None)
    req.status = RequestStatus.RUNNING
    scheduler._assign_waiting(req, tokens, query_tokens, step)
    scheduler.stats["loaded_kv_resumptions"] += 1
    controller = getattr(scheduler, "cache_controller", None)
    if controller is not None:
        controller.resumed(req)
    return True
