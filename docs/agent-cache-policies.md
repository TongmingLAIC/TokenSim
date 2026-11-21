# Programmable KV Cache policies

Quota uses the replay trace, play and lane identity to group related requests even when cache salting is disabled.
A global reservation counts distinct resident blocks and declared future growth across running, loading and loaded-waiting requests.
New admissions defer when this demand exceeds 92% of physical GPU slots.
Grouped growth uses the per-request decode window plus speculative lookahead; the layout-wide prefill scratch envelope cannot be reserved independently for every request.
The session-share timeout does not bypass this global reservation.
An empty active set permits progress; ordinary allocation still rejects an individually infeasible request.
This conservative reservation uses the request's declared output length, which is known to this simulator workload.
It does not guarantee zero preemptions under every grouped layout or speculative allocation pattern.

Quota-enabled controllers sort waiting requests by deferred state, then prefix hit tokens plus age credit and parent credit, then arrival time.
Age contributes 2048 tokens of credit per 30 seconds, and a parent contributes 1024 tokens.
The sorter uses current allocator state and cannot observe future completions.

## Cooperative movement

`all` combines evictable retention, quota and history-based tool-wait movement.
`forced_movement` combines retention and quota with a fixed 30-second movement window and no occupancy gate.
`enhanced_coop` adds queue-window estimation and an explicit one-shot child hint.

`all` and `enhanced_coop` require at least 80% occupied physical HBM before demotion.
Occupied space includes reclaimable cached pages; the allocator's allocated-block counter excludes them and cannot measure this pressure by itself.
Unknown or sub-0.5-second windows do not demote pages.
Completed cold writes remain required before reclaiming each GPU source.
The external reuse planner validates all cache groups at a common prefix boundary before import.

The enhanced window takes the larger of the historical admission gap and the estimated tool pause plus queue drain time.
Queue drain time uses observed admitted-to-completion service duration and current queue length divided by active concurrency.
Client resume and actual compute admission update separate exponentially weighted observations with weight 0.2.
The queue estimate is advisory and does not reserve capacity.

Only explicit `cache_one_shot=true` application metadata supplies a one-shot hint.
Only enhanced mode suppresses their private writes and reclaims their unreferenced cached slots.
Parent-associated content and live request/transfer references remain protected.
Multi-turn children keep ordinary cache behavior.
An unannotated single-turn child receives no hint; production traces and synthetic traces are labeled separately.

## Physical capacity control

`--kv_cache_memory_bytes` sets an explicit per-rank physical KV budget for ordinary serving.
It is mutually exclusive with a token-capacity override or an observed runtime-memory profile.
The budget cannot exceed hardware-derived available KV memory and must hold a null slot plus usable slots.
Grouped layouts round down to the physical slot stride and retain the original weight/workspace deductions.
The memory report records both the requested budget and the realized slot capacity.

## Waiting-queue scoring cost

Quota scoring constructs one physical ownership/growth view for a queue-scoring pass.
Candidates reuse that view and their existing prefix plan while the allocator state remains unchanged.
A fresh view is constructed for actual admissions and after every allocator-changing scheduling pass.
The scheduler checks its sequence and occupancy gates before scoring new work and skips new-work sorting while a loaded request resumes.
This removes repeated traversal of identical active-page sets without changing admission thresholds or ranking.

Enhanced one-shot checks reuse the parent-content set until a snapshot capture or release changes its revision.
This preserves the protected content while avoiding a full parent-prefix scan for every published child page.

## Independent cooperative components

Enhanced mode exposes three Boolean switches, all enabled by default.
`--cache_queue_window_enabled` controls admission-gap/queue-window estimation and client-versus-admission lifetime handling.
`--cache_child_write_admission` controls bypassing private one-shot child writes.
`--cache_child_reclamation` controls reclaiming their unreferenced cached GPU pages.
The corresponding `--no-...` options disable each component independently.
Disabling all three reproduces the `all` mechanism while retaining the enhanced controller label.
