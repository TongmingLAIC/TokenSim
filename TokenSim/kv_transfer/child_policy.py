"""Explicit one-shot child hints preserve live and parent-owned cache content."""


def parent_content(controller, manager):
    movement = controller.movement
    if movement is None:
        return set()
    revision = getattr(movement, "snapshot_revision", None)
    if revision is not None:
        cached = movement.parent_content_cache.get(id(manager))
        if cached is not None and cached[0] == revision:
            return cached[1]
    content = {
        tag
        for snapshot in movement.snapshots.values()
        if snapshot.manager is manager
        for _, tag in snapshot.blocks
    }
    if revision is not None:
        movement.parent_content_cache[id(manager)] = (revision, content)
    return content


def skip_write(controller, manager, block, req):
    return (
        controller.mode == "enhanced_coop"
        and getattr(controller, "child_write_admission", True)
        and getattr(req, "cache_one_shot", False)
        and block.block_hash not in parent_content(controller, manager)
    )


def reclaim_child(controller, scheduler, req, block_ids):
    manager = scheduler.block_manager
    if (
        controller.mode != "enhanced_coop"
        or not getattr(controller, "child_reclamation", True)
        or not manager.grouped_cache
        or not getattr(req, "cache_one_shot", False)
    ):
        return
    protected = parent_content(controller, manager)
    reclaimed = 0
    for slot in block_ids:
        block = manager.blocks[slot]
        if (
            block.ref_count
            or slot not in manager._free_cached
            or block.block_hash in protected
        ):
            continue
        del manager._free_cached[slot]
        manager._evict(block)
        manager._free_uncached.append(block)
        reclaimed += 1
    controller.stats["child_reclaimed_pages"] = (
        controller.stats.get("child_reclaimed_pages", 0) + reclaimed
    )
    controller.record("child_reclaimed", request_id=req.id, pages=reclaimed)
