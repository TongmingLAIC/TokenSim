"""Named DSpark buffers and draft graph descriptors."""

from math import ceil

from TokenSim.hardware.device import dtype_bytes


def persistent_allocations(model, tokens, max_num_seqs, drafts):
    if model.dspark is None or drafts == 0:
        return {}
    return dict(
        target_mtp_hidden=tokens
        * len(model.dspark.target_layer_ids)
        * model.hidden_size
        * 2,
        drafter_hidden=tokens * model.hidden_size * 2,
        dspark_draft_logits=(
            ceil(
                max_num_seqs
                * drafts
                * model.vocab_size
                * dtype_bytes(model.operator_dtype("lm_head"))
            )
            if model.dspark.draft_sample_method == "probabilistic"
            else 0
        ),
        dspark_confidence_probabilities=max_num_seqs * drafts * 4,
        dspark_anchor_indices=max_num_seqs * 8,
        dspark_step_columns=drafts * 4,
    )


def capture_rows(drafts, max_num_seqs, sample_from_anchor, target_sequences=None):
    width = drafts if sample_from_anchor else drafts + 1
    return sorted(
        {
            ceil(n * (drafts + 1) / width) * width
            for n in range(
                1,
                (target_sequences if target_sequences is not None else max_num_seqs)
                + 1,
            )
            if ceil(n * (drafts + 1) / width) <= max_num_seqs
        }
    )
