"""Observe uniform prefix sources without changing the native reuse plan."""


def observe_uniform_sources(req, gpu_tokens, cpu_tokens):
    gpu = min(gpu_tokens, max(0, req.context_len - 1))
    cpu = min(cpu_tokens, max(0, req.context_len - 1 - gpu))
    req.gpu_cached_context_tokens = gpu
    req.external_cached_context_tokens = cpu
    req.gpu_cached_prefill_tokens = min(req.prefill_len, gpu)
    req.external_cached_prefill_tokens = min(
        cpu, max(0, req.prefill_len - req.gpu_cached_prefill_tokens)
    )
    req.record_prefill_admission(
        req.gpu_cached_prefill_tokens, req.external_cached_prefill_tokens
    )
