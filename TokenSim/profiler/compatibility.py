"""Compare declared identities before evaluating duration differences."""

from pathlib import Path

from TokenSim.errors import ConfigurationError


def _expected_tp(simulated, observed, source, rank_map):
    layout = simulated.metadata.get("worker_parallelism")
    if not layout:
        return simulated.metadata.get("tensor_parallel_size")
    ranks = {
        str(rank_map.get(span.rank, span.rank))
        for span in observed.spans
        if span.source == source["path"]
    }
    if not ranks or not ranks <= layout.keys():
        return None
    values = {layout[rank]["tensor_parallel_size"] for rank in ranks}
    return next(iter(values)) if len(values) == 1 else None


def validate_inputs(simulated, observed, rank_map=None):
    checked, unavailable = [], []
    for source in observed.inputs:
        manifest = source["manifest"]
        config = manifest.get("model_config", {})
        actual = dict(
            model_id=Path(manifest.get("model", "")).name,
            dtype=manifest.get("dtype"),
            hidden_size=config.get("hidden_size"),
            num_layers=config.get("num_hidden_layers"),
            num_attention_heads=config.get("num_attention_heads"),
            tensor_parallel_size=manifest.get("tensor_parallel_size"),
        )
        for field, value in actual.items():
            expected = (
                _expected_tp(simulated, observed, source, rank_map or {})
                if field == "tensor_parallel_size"
                else simulated.metadata.get(field)
            )
            if expected is None or value in (None, ""):
                unavailable.append(dict(source=source["path"], field=field))
                continue
            if field == "model_id":
                expected, value = Path(str(expected)).name.lower(), str(value).lower()
            if field == "dtype":
                aliases = {"bfloat16": "bf16", "float16": "fp16"}
                expected, value = aliases.get(expected, expected), aliases.get(
                    value, value
                )
            if expected != value:
                raise ConfigurationError(
                    f"incompatible {field}: simulated={expected!r}, profiled={value!r}"
                )
            checked.append(dict(source=source["path"], field=field))
    return dict(checked=checked, unavailable=unavailable)
