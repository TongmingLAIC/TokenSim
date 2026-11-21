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
        controller.child_write_admission
        and getattr(req, "cache_one_shot", False)
        and block.block_hash not in parent_content(controller, manager)
    )


def reclaim_child(controller, scheduler, req, block_ids):
    manager = scheduler.block_manager
    if (
        not controller.child_reclamation
        or not manager.grouped_cache
        or not getattr(req, "cache_one_shot", False)
    ):
        return
    protected = parent_content(controller, manager)
    reclaimed = 0
    for slot in block_ids:
        block = manager.blocks[slot]
        if block.block_hash not in protected:
            reclaimed += manager.reclaim_cached_slot(slot, block.block_hash)
    controller.stats["child_reclaimed_pages"] = (
        controller.stats.get("child_reclaimed_pages", 0) + reclaimed
    )
    controller.record("child_reclaimed", request_id=req.id, pages=reclaimed)
