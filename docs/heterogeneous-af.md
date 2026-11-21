# Heterogeneous A/F execution groups

A worker group's `hardware` field accepts one device identifier for a homogeneous group or a list containing one device identifier per physical worker.
The list must contain exactly `num_workers` nonempty identifiers.
Worker indices, physical identities and execution-domain rank mappings retain their existing meanings.
The runtime compiler charges each worker's component weights and KV budget against its own device capacity, and the operator provider prices its kernels using that worker's device.
Existing string configurations retain their serialized representation.

Heterogeneous physical communication domains require an explicit catalog `topology` when devices have different scale-up links.
The physical placement builder takes the explicit topology's link classes and the declared communication-domain node inventory.
Communication service and contention use the explicit node aggregate bandwidth and latency settings.
Hardware generations must be assigned physically compatible nodes; cross-generation transfers use the declared inter-node fabric.
A logical A/F handoff on the same physical device uses zero interconnect service in both physical and flat-link modes.
The trace retains its logical payload while recording zero physical wire bytes.
Collective participants must remain unique physical devices.

Independent prefill and A/F decode pools select destination replicas in the decode pool's coordinate space.
The prefill source DP rank remains attached to the request while KV transfer is pending.
The request receives its selected decode DP rank when transfer completes.
The PD lifecycle export records `source_dp_rank` and `target_dp_rank` for A/F handoffs.
Ordinary P/D operation retains its shared DP-coordinate semantics.
