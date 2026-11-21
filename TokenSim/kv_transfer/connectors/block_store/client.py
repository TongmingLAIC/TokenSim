"""Uniform storage geometry, shared DRAM contribution, and prefix keys."""

from __future__ import annotations

from functools import cached_property
import json

from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_request import Request
from TokenSim.mooncake.config import parse_mooncake_config
from TokenSim.mooncake.metrics import MooncakeStats
from TokenSim.mooncake.pool_key import PoolKey, pool_keys_for_request
from TokenSim.mooncake.service import get_mooncake_service

from ..base import BaseKVConnector


class UniformStoreClient(BaseKVConnector):
    def __init__(
        self,
        config: KVTransferConfig,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__(config, cache_config)
        self.block_bytes = 1
        self.block_size = 1
        self.model_name = "unknown"
        self.rank_info = None
        if cache_config is not None:
            self.block_size = int(cache_config.block_size)
            self.block_bytes = int(
                cache_config.block_size * cache_config.size_per_token
            )
            self.model_name = cache_config.model
            self.rank_info = getattr(cache_config, "rank_info", None)
        self.options = parse_mooncake_config(config.kv_connector_extra_config)
        if self.options.aggregate_tensor_parallel:
            parallel = getattr(cache_config, "parallel_config", None)
            if parallel is None or parallel.pipeline_parallel_size != 1:
                raise ConfigurationError(
                    "Aggregate TP store objects require explicit PP1 replica geometry"
                )
            self.block_bytes *= parallel.tensor_parallel_size
        if self.options.replica_num != 1:
            raise ConfigurationError(
                "uniform Mooncake storage supports a single replica"
            )
        if self.options.transfer_overlap and not self.async_offload:
            raise ConfigurationError(
                "uniform metadata timing does not support transfer_overlap; it waits for I/O completion"
            )
        self._bind_service()
        # "mooncake" mirrors the reference vLLM MooncakeStoreConnector
        # (ref/vllm .../kv_connector/v1/mooncake/store/); "every_step" keeps
        # the historical TokenSim behavior.
        self._aligned = self.options.save_policy == "mooncake"
        # Per-request key chain and its string form, both computed exactly once.
        self._keys_cache: dict[int, tuple[list[PoolKey], list[str]]] = {}

    def _bind_service(self, owner=None, contributor=None):
        self.service = get_mooncake_service(
            self.options,
            block_bytes=self.block_bytes,
            owner=owner,
            contributor=contributor,
        )
        self.mooncake_stats = MooncakeStats()
        self.service.store.register_storage_stats(self.mooncake_stats)

    def bind_runtime(self, worker):
        self._bind_service(worker.env, worker)
        mode = "queued" if self.async_offload else "metadata"
        existing = getattr(self.service, "uniform_io_mode", mode)
        if existing != mode:
            raise ConfigurationError(
                "A shared store cannot mix queued and metadata-only uniform clients"
            )
        self.service.uniform_io_mode = mode

    def on_request_released(self, req: Request) -> None:
        self._keys_cache.pop(req.id, None)

    def _keys_for_request(self, req: Request) -> list[PoolKey]:
        return self._request_key_bundle(req)[0]

    def _key_strings_for_request(self, req: Request) -> list[str]:
        return self._request_key_bundle(req)[1]

    @cached_property
    def store_namespace(self):
        cache = self.cache_config
        model = getattr(cache, "model_spec", None)
        parallel = getattr(cache, "parallel_config", None)
        shape = [
            self.block_size,
            self.block_bytes,
            getattr(parallel, "tensor_parallel_size", 1),
            getattr(parallel, "pipeline_parallel_size", 1),
            getattr(model, "kv_cache_dtype", None),
            getattr(model, "dtype", None),
            model.cache_signature() if model is not None else None,
        ]
        return "@layout:" + json.dumps(shape, separators=(",", ":"))

    def _request_key_bundle(self, req: Request) -> tuple[list[PoolKey], list[str]]:
        bundle = self._keys_cache.get(req.id)
        if bundle is None:
            keys = pool_keys_for_request(
                req,
                model_name=self.model_name,
                rank_info=self.rank_info,
                store_namespace=self.store_namespace,
                cache_prefix=self.options.cache_prefix,
                tenant_id=self.options.tenant_id,
                group_id=int(self.options.group_id),
                pcp_rank=int(self.options.pcp_rank),
                dcp_rank=int(self.options.dcp_rank),
            )
            bundle = (keys, [key.to_string() for key in keys])
            self._keys_cache[req.id] = bundle
        return bundle
