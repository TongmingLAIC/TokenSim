"""Rank-local model work with shared-F deduplication and physical denominators."""

from collections import defaultdict
import json

from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.workload import work_for_table
from TokenSim.operator_data.analytical import analytical_model_for
from TokenSim.trace.occupancy import unique_executions


def ledger(events, hardware, inventory, *, start, end, gated=True):
    if end <= start or not inventory or min(inventory.values()) < 1:
        raise ConfigurationError(
            "FLOP accounting requires a positive window and physical inventory"
        )
    totals = defaultdict(float)
    normalized = defaultdict(float)
    components = defaultdict(
        lambda: dict(
            invocations=0,
            flops=0.0,
            peak_seconds=0.0,
            service_seconds=0.0,
            analytical_compute_seconds=0.0,
            analytical_memory_seconds=0.0,
            analytical_overhead_seconds=0.0,
            memory_bound_invocations=0,
        )
    )
    analysis_cache = {}
    logical_count = 0

    def counted_events():
        nonlocal logical_count
        for event in events:
            logical_count += int(event.event_type == "compute")
            yield event

    physical_count = 0
    for event in unique_executions(counted_events()):
        if (
            event.event_type != "compute"
            or event.end_at <= start
            or event.start_at >= end
        ):
            continue
        if event.start_at < start - 1e-12 or event.end_at > end + 1e-12:
            raise ConfigurationError(
                "A partial compute event cannot provide exact completed-work MFU"
            )
        physical_count += 1
        device_id = event.details["device_id"]
        device = hardware.device(device_id)
        work_components = event.details.get("components")
        if work_components is None:
            work_components = event.details.get("work_components")
            if work_components is None:
                if event.operation == "step_overhead":
                    continue
                work_components = [
                    dict(
                        name=event.operation,
                        table=event.details.get("table"),
                        key=event.details.get("shape", {}),
                    )
                ]
            work_components = [
                dict(c, latency_us=event.service_duration * 1e6 / len(work_components))
                for c in work_components
            ]
        for component in work_components:
            if "work_flops_by_dtype" in component:
                values = component["work_flops_by_dtype"]
                charges = {
                    p: f / device.peak_compute_for(p).value for p, f in values.items()
                }
                total_charge = sum(charges.values())
                for precision, flops in values.items():
                    if flops < 0:
                        raise ConfigurationError(
                            "compressed useful FLOPs must be nonnegative"
                        )
                    fraction = charges[precision] / total_charge if total_charge else 0
                    totals[device_id] += flops
                    normalized[device_id] += charges[precision]
                    identity = (
                        device_id,
                        component["name"],
                        component["table"],
                        precision,
                    )
                    row = components[identity]
                    row["invocations"] += 1
                    row["flops"] += flops
                    row["peak_seconds"] += charges[precision]
                    row["service_seconds"] += component["latency_us"] * 1e-6 * fraction
                    bounds = component["analytical_bounds"]
                    row["analytical_compute_seconds"] += (
                        bounds["compute_us"] * 1e-6 * fraction
                    )
                    row["analytical_memory_seconds"] += (
                        bounds["memory_us"] * 1e-6 * fraction
                    )
                    row["memory_bound_invocations"] += int(
                        bounds["memory_us"] >= bounds["compute_us"]
                    )
                continue
            key = component["key"]
            work = work_for_table(component["table"], key, gated=gated)
            dtype = key.get("attn_dtype", key.get("dtype"))
            if dtype is None:
                if work.flops == 0:
                    continue
                raise ConfigurationError(
                    f"Operator lacks a compute precision: {component['table']}"
                )
            peak = float(device.peak_compute_for(dtype).value)
            totals[device_id] += work.flops
            normalized[device_id] += work.flops / peak
            identity = (device_id, component["name"], component["table"], str(dtype))
            row = components[identity]
            row["invocations"] += 1
            row["flops"] += work.flops
            row["peak_seconds"] += work.flops / peak
            row["service_seconds"] += component["latency_us"] * 1e-6
            fingerprint = (
                device_id,
                component["table"],
                json.dumps(key, sort_keys=True),
            )
            if fingerprint not in analysis_cache:
                analysis_cache[fingerprint] = analytical_model_for(
                    device.family
                ).estimate(component["table"], key, device, gated=gated)
            analysis = analysis_cache[fingerprint]
            row["analytical_compute_seconds"] += analysis.compute_time_us * 1e-6
            row["analytical_memory_seconds"] += analysis.memory_time_us * 1e-6
            row["analytical_overhead_seconds"] += analysis.overhead_us * 1e-6
            row["memory_bound_invocations"] += int(
                analysis.memory_time_us >= analysis.compute_time_us
            )
    duration = end - start
    return dict(
        start_s=start,
        end_s=end,
        duration_s=duration,
        physical_inventory=inventory,
        logical_compute_spans=logical_count,
        physical_compute_invocations=physical_count,
        useful_flops=dict(totals),
        peak_normalized_device_seconds=dict(normalized),
        system_model_utilization=sum(normalized.values())
        / (sum(inventory.values()) * duration),
        device_model_utilization={
            d: normalized[d] / (count * duration) for d, count in inventory.items()
        },
        components=[
            dict(device=d, name=n, table=t, compute_dtype=p, **values)
            for (d, n, t, p), values in sorted(components.items())
        ],
        definition="Sum canonical rank-local useful FLOPs / matching precision peak, divided by allocated physical device-seconds; shared executions counted once",
    )


def write_ledger(path, events, hardware, inventory, *, start, end, gated=True):
    result = ledger(events, hardware, inventory, start=start, end=end, gated=gated)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result
