"""One Mooncake connector interface with explicit cache-layout configuration."""

from TokenSim.errors import ConfigurationError
from TokenSim.mooncake.config import parse_mooncake_config

from .base import BaseKVConnector
from .block_store.core import BlockStore
from .store_pages import PageStore


class MooncakeStoreConnector(BaseKVConnector):
    name = "MooncakeStoreConnector"
    supports_grouped_cache = True

    def __new__(cls, config, cache_config=None):
        options = parse_mooncake_config(config.kv_connector_extra_config)
        grouped = getattr(cache_config, "hybrid_layout", None) is not None
        if (options.cache_layout == "grouped") != grouped:
            raise ConfigurationError(
                "Mooncake cache_layout must match the KV cache: "
                "grouped KV requires cache_layout=grouped; uniform KV requires uniform"
            )
        storage = PageStore if options.cache_layout == "grouped" else BlockStore
        return storage(config, cache_config)
