"""DSpark context insertion, one block-parallel draft and sequential Markov heads."""

from collections import Counter
import math

from TokenSim.errors import ConfigurationError
from TokenSim.latency.base import query_tokens, roofline_latency_us
from TokenSim.latency.moe import MegaMoEEstimator
from TokenSim.latency.speculative import SpeculativeEstimator
from TokenSim.llm.forward import ForwardPass, RankShape
from TokenSim.block.hybrid.dspark_memory import capture_rows


class DSparkEstimator(SpeculativeEstimator):
    def __init__(self, backend, *, max_num_seqs):
        if backend.model.dspark is None or backend.tp != 1:
            raise ConfigurationError(
                "DSpark estimation requires explicit draft geometry and TP1"
            )
        super().__init__(backend, max_num_seqs=max_num_seqs)
        self.draft = self.model.dspark
        self.query_width = self.k if self.draft.sample_from_anchor else self.k + 1
        self.initial_moe = (
            MegaMoEEstimator(
                backend,
                backend.moe_execution,
                expected_query_width=self.query_width,
                max_num_seqs=max_num_seqs,
            )
            if backend.native_moe
            else None
        )
        self.decode_moe = None
        # The measured D configurations capture every target request count.
        # The native draft dispatcher rounds those token counts to draft width.
        self.capture_rows = capture_rows(
            self.k, max_num_seqs, self.draft.sample_from_anchor
        )

    def _source_step(self, requests):
        if self.step is not None:
            return self.step
        if self.backend.parallel_config.data_parallel_size != 1:
            raise ConfigurationError(
                "DSpark needs the synchronized DP request snapshot"
            )
        return ForwardPass(
            {
                self.backend.rank_info.dp_rank: RankShape(
                    sum(query_tokens(r) for r in requests), None, True, len(requests)
                )
            }
        )

    def pass_shapes(self, requests):
        source = self._source_step(requests)
        counts = {
            rank: self._request_count(
                shape,
                own_count=(
                    len(requests) if rank == self.backend.rank_info.dp_rank else None
                ),
            )
            for rank, shape in source.ranks.items()
        }
        maximum = max(counts.values()) * self.query_width
        padded = next((rows for rows in self.capture_rows if rows >= maximum), None)
        shapes = {}
        for rank, shape in source.ranks.items():
            valid = (
                counts[rank] if shape.valid_requests is None else shape.valid_requests
            )
            rows = counts[rank] * self.query_width if padded is None else padded
            shapes[rank] = RankShape(
                rows,
                self.query_width,
                False,
                rows // self.query_width,
                valid * self.query_width,
                valid,
            )
        step = source.derived(shapes)
        rows = shapes[self.backend.rank_info.dp_rank].tokens
        return [(rows, rows, step, padded is not None)]

    def _attention(self, requests):
        pairs = 0
        width = self.query_width
        for req in requests:
            prefix = req.num_computed_tokens + req.step_tokens
            pairs += width * (
                min(prefix, max(0, self.model.sliding_window - 1)) + width
            )
        self.backend.stats.record("dspark_noncausal_swa", "analytical")
        return roofline_latency_us(
            self.backend.device,
            pairs * self.model.head_dim,
            4 * self.backend.heads_local * self.model.head_dim * pairs,
            dtype=self.model.activation_dtype,
            compute_efficiency=self.backend.device.analytical["attention_mfu"],
        )

    def estimate(self, requests):
        if not requests:
            return Counter()
        b, h = self.backend, self.model.hidden_size
        draft_layers = self.draft.num_layers
        context_rows = b._target_forward_rows(
            sum(query_tokens(r) for r in requests), requests
        )
        values = Counter()
        values["elementwise"] += self._stream_us(
            "dspark_hidden_gather",
            context_rows * h * len(self.draft.target_layer_ids) * 2,
        )
        values["gemm"] += b.gemm(
            context_rows,
            h,
            h * len(self.draft.target_layer_ids),
            dtype=self.model.operator_dtype("attention_gemm"),
        )
        values["elementwise"] += b.elementwise("rmsnorm", context_rows, h)
        for _ in range(draft_layers):
            values["gemm"] += b.gemm(
                context_rows,
                self.spec.q_lora_rank + self.model.head_dim,
                h,
                dtype=self.model.operator_dtype("attention_gemm"),
            )
            values["elementwise"] += b.elementwise(
                "rmsnorm", context_rows, self.model.head_dim
            )
            values["elementwise"] += self._stream_us(
                "dspark_context_rope_quant_insert",
                context_rows * self.model.head_dim * 5,
            )
        rows, _, step, graph = self.pass_shapes(requests)[0]
        values["elementwise"] += self._stream_us(
            "dspark_query_embedding", rows * h * 2 * (1 + self.spec.hc_mult)
        )
        attention = self._attention(requests)
        for _ in range(draft_layers):
            values["gemm"] += sum(
                b.query(p.table, p.key(rows)).latency_us for p in self.projections
            )
            values["gemm"] += b.gemm(
                rows,
                self.model.moe.num_experts,
                h,
                dtype=self.model.operator_dtype("router"),
            )
            values["attention"] += attention
            values["elementwise"] += self._mhc(rows) + 2 * b.elementwise(
                "rmsnorm", rows, h
            )
            values["moe"] += self._moe(rows, step, True)
            if b.ep_enabled and self.initial_moe is None and b.communicator:
                values["comm"] += 1e6 * b.communicator.estimate_ep_all2all(
                    rows * self.model.moe.num_experts_per_tok * h * 2,
                    count=1,
                    num_tokens=rows,
                    hidden_size=h,
                    top_k=self.model.moe.num_experts_per_tok,
                    num_experts=self.model.moe.num_experts,
                    activation_bytes=2,
                )
        request_rows = rows // self.query_width
        sample_rows = request_rows * self.k
        values["lm_head"] += self.hc_head.estimate(rows)
        values["lm_head"] += b.elementwise("rmsnorm", sample_rows, h)
        values["lm_head"] += b.gemm(
            sample_rows, b.vocab_local, h, dtype=self.model.operator_dtype("lm_head")
        )
        for _ in range(self.k):
            values["elementwise"] += self._stream_us(
                "dspark_markov_embedding", request_rows * self.draft.markov_rank * 2
            )
            values["lm_head"] += b.gemm(
                request_rows,
                self.model.vocab_size,
                self.draft.markov_rank,
                dtype=self.model.operator_dtype("lm_head"),
            )
            if self.draft.draft_sample_method == "greedy":
                values["elementwise"] += b.elementwise(
                    "argmax", request_rows, self.model.vocab_size
                )
            else:
                values["elementwise"] += self._stream_us(
                    "dspark_markov_sampling",
                    request_rows * self.model.vocab_size * 12,
                    flops=request_rows * self.model.vocab_size * 8,
                )
        self.counts.update(
            target_steps=1,
            context_rows=context_rows,
            context_kv_layer_rows=context_rows * draft_layers,
            block_forwards=1,
            backbone_layer_rows=rows * draft_layers,
            query_rows=rows,
            logits_rows=sample_rows,
            markov_steps=self.k,
            markov_rows=request_rows * self.k,
            graph_passes=int(graph),
            eager_passes=int(not graph),
        )
        for name, value in values.items():
            self.component_seconds[name] += value * 1e-6
        return values

    def describe(self):
        return dict(
            method="dspark",
            K=self.k,
            draft=self.draft.to_dict(),
            passes="One eager target-hidden projection and per-layer context-KV insertion, then one block-parallel draft backbone",
            sampling=(
                "K sequential full-vocabulary Markov projections; "
                + (
                    "greedy selection"
                    if self.draft.draft_sample_method == "greedy"
                    else "probabilistic draws"
                )
                + "; adaptive verification disabled"
            ),
            query_width=self.query_width,
            capture_token_rows=self.capture_rows,
            attention="Noncausal query block plus the retained SWA context; analytical single-latent core",
            analytical_components="Context insertion, noncausal SWA, Markov sampling and state movement; primitive GEMM and optional native mHC/MoE use the selected operator backend",
        )
