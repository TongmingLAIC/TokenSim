# AF service variability and controlled peer synchronization

`af.jitter_model` adds synthetic service variability for sensitivity analysis.
Its defaults preserve the original provider estimates exactly.
The `enabled`, `compute_cv`, `network_cv` and `seed` fields select the perturbation model.
Both coefficients of variation are finite values in [0, 2], and the seed is a nonnegative integer.
Each request receives a mean-one lognormal service multiplier, keyed by seed, layer, operation, phase and context.
The multiplier for an invocation is the mean of its request multipliers, including all routing-edge requests of a merged expert invocation.
This model represents synthetic variation in per-request work and imposes no maximum-over-batch penalty.
Compute and network streams are separate, and submission order and owner placement do not enter their keys.
Communication phase identities additionally distinguish collective phases.
The model scales device or communication service time; native resource queues still produce contention.
It introduces no calibrated hardware claims or independent congestion penalty.
Trace details retain the original service duration and applied multiplier.

`af.peer_layer_barrier=true` adds a controlled barrier at each model layer entry across all configured A peers for matching submission rounds and microbatch indices.
Routing location, expert combination and tensor return formats retain their selected protocol semantics.
The switch defaults to false and is separate from the `vllm_afd` protocol.
All peers must submit the same number and phases of microbatches in each round.
A shorter final microbatch is supported when the number and phases match.
Incompatible peer microbatch counts are rejected, and an absent peer leaves the barrier pending.
Cancellation or failure aborts all peer waiters; a new executor is required after abort.
The barrier releases once every peer has arrived and records queue wait as a synchronize event.
Credit limits retain their default per-A-owner, per-iteration scope.

Decode-only studies report decode-window goodput and MFU, and do not imply measured end-to-end serving performance.

## Rank variation, packet tails and receive buffers

`rank_compute_cv` adds a mean-one multiplicative disturbance per physical compute rank, layer, operation and context step.
This component remains correlated across requests sharing that invocation; increasing batch size does not average it away.
`tp_cv` perturbs collective service separately from external A/F transfers.
Actual rank dependencies determine how slow computation and TP completion delay a cohort.

`af.transport_effects` describes optional synthetic stress scenarios and defaults to disabled.
The packet size, per-packet tail probability, Lomax tail scale and shape determine completion of the last packet in each transferred tensor.
A shape greater than one gives a finite conditional packet-tail mean.
The maximum packet delay is sampled by CDF inversion, with per-request identities and exact total packet count.
Large messages can encounter a late packet more frequently without an imposed batch-size multiplier.

The switch-buffer byte threshold and retry duration model an additional drop-tail retry scenario.
Outstanding queued and flying bytes above the threshold require additional retry rounds.
This is an explicit synthetic assumption, and its zero-retry control separates it from ordinary wire queueing.
Packet flight and retries occur after native wire service and do not retain endpoint serialization resources.
Native trace metadata identifies the actual resource-occupancy interval.

A finite number of receive-cohort slots is reserved before sending and held until every F compute consumer of that layer has completed.
The input slot is released before the F-to-A return flight or A-side residual work finishes.
This reservation prevents fast later arrivals from consuming every slot needed by an earlier missing cohort.
With ordered receive enabled, a message waits only until earlier posted messages on the same F receive stream have arrived.
The receive-order gate advances immediately at message arrival and is independent of the cohort slot lifetime.
Ready receive permits it to proceed independently.
Both modes use the same wire, packet-tail and buffer parameters.
Packet timestamps, retry rounds, receiver admission and ordering waits are exported separately.
Ordering is a tested protocol assumption, and is not attributed to every AFD implementation.

Messages join their receive stream only after their cohort obtains input capacity.
A preposted message without capacity could otherwise wait for a reserved cohort's later fragment while that fragment waits behind the unadmitted message.
The delayed-fragment regression explicitly rules out this circular wait.
Cancellation removes pending receiver reservations and their reserve processes.
The basic executor and whole-model executor both release input capacity after their F consumers finish.
Packet records include the receive-stream identity, post index and earlier messages responsible for an ordering wait.

## TP participant variability

`tp_rank_cv` adds synthetic communication variation per participant of a TP all-reduce, all-gather or reduce-scatter.
Its default is zero and preserves all earlier numerical behavior.
Each participant receives a mean-one lognormal residual-service multiplier, and collective completion follows the slowest participant.
The fixed propagation term is retained.
This explicitly sampled maximum represents extra rank heterogeneity relative to the homogeneous base service.
Its expected synchronization cost increases with participant count; it is not a fixed p99 factor applied to every batch.
Participant sampling is independent of request batch size and does not average away at high concurrency.
`tp_cv` retains its earlier request-averaged collective perturbation for reproduction of the initial controls.
The TP-specific study disables `tp_cv` and uses `tp_rank_cv` separately from `rank_compute_cv`.
Both flat collectives and physical collective phases supply their physical participant identities.

## Credits across cohort continuations

`af.max_inflight_scope` defaults to `iteration`, preserving the existing admission window.
Selecting `owner` shares the configured credit count across every overlapping iteration of one logical A owner.
Each admitted microbatch holds one credit until its complete model step finishes, including transfers.
Cancellation releases the credit through the existing reservation context.
A caller can submit independent sequential decode continuations for different cohorts.
A fast cohort may begin its following token while another cohort of the same A owner is still completing an older token.
Owner-scoped credits retain the physical in-flight bound across those continuations.
