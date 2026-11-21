"""Compatibility imports for storage.native."""

from TokenSim.kv_transfer.storage.native import (
    NativePageStore,
    QueuedKVLink,
    StoredPage,
    validate_external_stores,
)

__all__ = ["NativePageStore", "QueuedKVLink", "StoredPage", "validate_external_stores"]
