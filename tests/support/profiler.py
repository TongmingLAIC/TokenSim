"""Small matching simulator and GPU timeline files."""

import gzip
import json
from pathlib import Path


def fixtures(directory, *, durations=(5, 4)):
    directory = Path(directory)
    simulated = directory / "simulated.json"
    profile = directory / "profile_tp0.json.gz"
    sim = []
    observed = [
        dict(
            ph="X",
            name="tokensim.forward",
            cat="user_annotation",
            pid=1,
            tid=1,
            ts=1000,
            dur=100,
        )
    ]
    for index, (name, start, duration) in enumerate(
        zip(("qkv", "output"), (0, 5), durations)
    ):
        sim.append(
            dict(
                ph="X",
                name=name,
                cat="compute",
                pid=1,
                tid=1,
                ts=start,
                dur=5 if index == 0 else 4,
                args=dict(
                    node_id=f"forward:0:0:rank:0:{index}",
                    operation=name,
                    worker_id=0,
                    iteration_id=0,
                    layer_id=0,
                    phase="prefill",
                    token_count=1,
                ),
            )
        )
        observed.extend(
            [
                dict(
                    ph="X",
                    name=f"tokensim.layer.0.{name}",
                    cat="user_annotation",
                    pid=1,
                    tid=1,
                    ts=1010 + index * 20,
                    dur=10,
                ),
                dict(
                    ph="X",
                    name="aten::linear",
                    cat="cpu_op",
                    pid=1,
                    tid=1,
                    ts=1011 + index * 20,
                    dur=2,
                    args={"External id": index + 1},
                ),
                dict(
                    ph="X",
                    name="gemm_kernel",
                    cat="kernel",
                    pid=0,
                    tid=7,
                    ts=2000 + index * 7,
                    dur=duration,
                    args={"External id": index + 1, "device": 0, "stream": 7},
                ),
            ]
        )
    simulated.write_text(
        json.dumps(dict(traceEvents=sim, metadata={"model_id": "Fixture"}))
    )
    with gzip.open(profile, "wt") as stream:
        json.dump(dict(traceEvents=observed), stream)
    (directory / "manifest.json").write_text(
        json.dumps(dict(model="Fixture", batch_size=1, input_tokens=1))
    )
    return simulated, profile
