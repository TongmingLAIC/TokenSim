"""DeepSeek DSpark draft geometry recovered from the measured container."""

from dataclasses import asdict, dataclass

from TokenSim.config.validation import integer, boolean
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class DSparkConfig:
    num_layers: int = 3
    target_layer_ids: tuple[int, ...] = (58, 59, 60)
    markov_rank: int = 512
    confidence_head: bool = True
    sample_from_anchor: bool = True
    draft_sample_method: str = "probabilistic"

    def __post_init__(self):
        integer(self.num_layers, "DSpark layers", minimum=1)
        integer(self.markov_rank, "DSpark Markov rank", minimum=1)
        boolean(self.confidence_head, "DSpark confidence head")
        boolean(self.sample_from_anchor, "DSpark anchor sampling")
        if self.draft_sample_method not in {"greedy", "probabilistic"}:
            raise ConfigurationError("Unknown DSpark draft sampling method")
        if not self.target_layer_ids:
            raise ConfigurationError(
                "DSpark needs target hidden-state layer identifiers"
            )
        for layer in self.target_layer_ids:
            integer(layer, "DSpark target layer")
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ConfigurationError("DSpark target layers must be unique")
        object.__setattr__(self, "target_layer_ids", tuple(self.target_layer_ids))

    @classmethod
    def from_value(cls, value, model=None):
        if value is None:
            return None
        if isinstance(value, dict):
            try:
                value = cls(**value)
            except TypeError as error:
                raise ConfigurationError(f"Invalid DSpark geometry: {error}") from error
        if not isinstance(value, cls):
            raise ConfigurationError("DSpark geometry must be an object")
        if model is not None and (
            model.components is None
            or model.components.num_nextn_predict_layers != value.num_layers
            or max(value.target_layer_ids) >= model.num_layers
        ):
            raise ConfigurationError(
                "DSpark target and draft layers disagree with the model geometry"
            )
        return value

    def to_dict(self):
        return asdict(self)


def add_dspark_weights(model, add, hc_head):
    draft = model.dspark
    h, vocab, rank = model.hidden_size, model.vocab_size, draft.markov_rank
    add(
        "dspark.main_proj",
        "dspark_context",
        (h, h * len(draft.target_layer_ids)),
        "attention_gemm",
    )
    add("dspark.main_norm", "norm", (h,), "norm")
    add("dspark.final_norm", "norm", (h,), "norm")
    hc_head("dspark.hc_head")
    add("dspark.markov_w1", "dspark_markov", (vocab, rank), "embedding")
    add("dspark.markov_w2", "dspark_markov", (vocab, rank), "lm_head")
    if draft.confidence_head:
        add(
            "dspark.confidence.weight",
            "dspark_confidence",
            (1, h + rank),
            "lm_head",
            storage_dtype="fp32",
        )
