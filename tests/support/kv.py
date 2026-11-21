"""Small grouped cache geometry with full and windowed attention."""

from TokenSim.block.hybrid.kv_layout import CacheGroup, CacheLayerPage, HybridKVLayout


def layout(eagle=False):
    return HybridKVLayout(
        groups=(
            CacheGroup(0, (CacheLayerPage("full", "full", 16, 8, alignment=1),)),
            CacheGroup(
                1,
                (
                    CacheLayerPage(
                        "window", "window", 4, 16, window_tokens=8, alignment=1
                    ),
                ),
                eagle=eagle,
            ),
        ),
        logical_block_tokens=16,
        physical_slot_bytes=128,
        retention_interval=32,
        speculative_tokens=1 if eagle else 0,
        max_in_flight_tokens=8,
        max_model_len=256,
    )
