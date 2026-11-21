# Programmable L2/L3 KV cache

`TieredKVConnector` is a TokenSim connector for grouped, immutable, content-addressed KV pages.
It models policy choices with CPU simulation and uses existing operator timing inputs.

## Capacity and transfer contracts

L1 is the worker GPU allocator.
Each rank has an independent L2 Host DRAM budget, including a reserved staging area.
All compatible clients in one simulation environment share one finite L3 capacity and one aggregate network budget.
Equal model, cache-group, TP/PP and shard scopes share L3 objects; the content key retains the complete salted prefix identity.
Use this connector for workers sharing one modeled storage service and network budget.
A multi-node deployment needs explicit per-node network and staging resources before this connector can represent it.

L2 and GPU–host DMA use the physical grouped-page stride inherited from SimpleCPU packing.
L3 stores actual grouped payload bytes, with independent finite capacity.
The result records this packing difference, both capacities and the 64 MiB per-rank staging reservation.
Remote transfers serialize through the finite staging buffer; write-around releases GPU DMA references after the GPU-to-host copy while staging remains occupied until the remote write finishes.
A local source page stays pinned throughout a write-back transfer.
An external destination becomes visible only after every required transfer and backend operation completes.
The existing all-group reuse planner determines the common reusable prefix boundary.
Cancelled GPU imports retain their DMA ownership until completion and do not publish request-visible pages.

The network has independent directional queues shared by all clients.
A single nonpreemptive backend queue handles reads and writes and can prioritize queued foreground reads.
The DRAM and SSD bandwidths and fixed latencies are scenario assumptions, recorded in each connector configuration.
They are not new device measurements or calibrated SSD predictions.

## Write and promotion semantics

| Policy | Trigger and source ownership |
| --- | --- |
| `write_through` | After the L2 copy finishes, enqueue a remote copy and pin its L2 source until completion. |
| `write_back` | Preserve L2-only pages; copy missing remote pages at the high watermark or before L2 eviction. |
| `write_around` | Bypass L2 cache admission, stage GPU data in the bounded Host DRAM buffer and then save L3. |
| `selective_write_back` | Apply write-back while excluding pages computed under explicit one-shot child hints from background persistence. |

The watermark defaults are 0.8 high and 0.6 low.
Foreground requests do not wait for remote durability; allocation can still wait for outstanding physical DMA references.
A write-back eviction can defer while its remote copy is incomplete.
If all candidate pages are pinned or veto eviction, the optional incoming cache write is refused and counted.
This refusal preserves existing copies and can cause later recomputation.
Write-through also records remote admission failures when the finite L3 cannot accept another pending copy.
Reading an L3 object can promote it to L2; `promote_reads=false` disables that admission independently of the write trigger.

Selective persistence consumes explicit application metadata and observes no future turn count.
An unannotated production trace gives selective write-back the same information as write-back.
A later ordinary publication removes the one-shot exclusion for a locally known object.
Selective persistence can discard reusable content after a mistaken application hint; KV recomputation remains the recovery path.
Session completion never invalidates a shared physical object belonging to another request.

## Observation, accepted-job drain and persistence audit

Every client captures its end-of-workload state before any client begins tail flushing.
This state includes L3-missing local payload bytes and completed remote read/write bytes.
After draining already accepted jobs, the CLI runs a separate persistence audit that makes one additional save attempt for each eligible remaining L2 object.
The final cache window records the audit traffic and remaining missing copies separately from policy traffic.
Finite L3 capacity can evict previously saved objects during the forced audit.
The record therefore reports both completed bytes and remaining missing bytes, without promising that all local content fits remotely.
Selective exclusions remain excluded during the audit.
Reported request throughput uses successful request boundaries and excludes this additional audit time.
The audit is an imposed persistence condition, not a forecast that ordinary write-back must eventually save every L2 object.
A small finite L3 cannot retain the entire larger L2, so audit churn does not establish ordinary steady-state write amplification.
Fixed-window throughput and SLO goodput count only completions inside the prescribed issue window.

## Profiling counter boundaries

The engine captures cumulative link and L3 counters at the profiling reset, preserving the warmed cache and in-flight physical work.
The exporter subtracts this exact baseline from end-of-workload and post-audit counters.
Work still pending at the boundary can consume resources during profiling; the baseline records its pending state.
Historical warmup runs without a baseline retain their lifetime counters and leave profiling-directional fields empty.
The exporter never substitutes zero for an unavailable profiling counter.
Peak storage fields are lifetime gauges and can include warmup and the forced persistence audit.

End-of-workload coverage counts distinct ready content objects across L1, L2 and L3 using model/layout/shard namespaces.
It reports replicated L2 payload, unique layer payload, L2/L3 overlap and the all-tier union.
These object counts do not assert that every stored object belongs to a complete reusable context; the grouped prefix planner and effective hit metrics establish that condition.

## Write-back victim ownership and bounded admission

An L2 source lease for write-back increments its physical reference count without refreshing its LRU age.
A completed write-back therefore leaves the intended old victim eligible for eviction when its remote copy becomes valid.

Native storage maintains an exact count of unpinned pages and validates that count after drain.
An all-pinned store rejects an allocation without scanning its LRU.
After an eviction veto, the allocator traverses the remaining candidate order once.
Tiered admission coalesces repeated failures while local/remote material state and capacities remain unchanged.
Reference, readiness and allocation changes invalidate that failed-state memo, so a completed remote copy permits a fresh attempt.
The memo changes redundant planning-attempt counts, not the physical capacity or completion contract.

A lazily initialized eviction index orders eligible pages by their actual LRU touch sequence.
It avoids repeatedly traversing a pinned LRU prefix under write-back pressure.
Read touches and reference changes invalidate stale index entries; bounded compaction limits metadata growth.
Releasing writes out of order preserves the original access age.

## Hysteretic write-back watermarks

`watermark_eviction=true` is the default write-back contract.
At the high watermark, the controller selects enough eligible LRU victims to reduce L2 toward the low watermark.
Pages with a valid L3 copy can be reclaimed immediately.
Other victims remain pinned until their L3 copy completes, then lose their L2 copy if no new read touched them.
Pending eviction bytes count toward the low-watermark target, preventing every new GPU batch from scheduling the same reclaim amount again.
An intervening read or live reference preserves the local page and cancels that stale reclaim intent.
Selective write-back can discard explicitly marked one-shot objects without L3 persistence.
The strict allocator continues to enforce actual resident capacity while reclamation is pending.

`watermark_eviction=false` retains the diagnostic copy-only operation.
Copying a page without freeing L2 does not lower physical occupancy and can repeatedly trigger background copies when L3 is smaller than the retained working set.
Through and around rows retain their unchanged `v2` implementations.
