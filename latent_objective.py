"""Item-normalized latent-SID objectives for one-stage Tokenizer + GR."""

import torch
import torch.nn.functional as F


def topk_factorized_paths(assignment_probabilities, beam_size):
    """Return the highest-probability paths of a factorized SID distribution."""
    if assignment_probabilities.dim() < 3:
        raise ValueError("assignment probabilities must end in [levels, codes]")
    if beam_size < 1:
        raise ValueError("beam_size must be positive")
    if (assignment_probabilities < 0).any():
        raise ValueError("assignment probabilities must be non-negative")
    leading_shape = assignment_probabilities.shape[:-2]
    levels, codes = assignment_probabilities.shape[-2:]
    flat = assignment_probabilities.reshape(-1, levels, codes)
    tiny = torch.finfo(flat.dtype).tiny
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(tiny)
    log_q = flat.clamp_min(tiny).log()

    path_scores = torch.zeros(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)
    paths = torch.empty(
        flat.shape[0], 1, 0, device=flat.device, dtype=torch.long
    )
    for level in range(levels):
        expanded = path_scores.unsqueeze(-1) + log_q[:, level].unsqueeze(1)
        expanded = expanded.reshape(flat.shape[0], -1)
        keep = min(int(beam_size), expanded.shape[1])
        path_scores, flat_indices = expanded.topk(keep, dim=-1)
        parent_indices = torch.div(flat_indices, codes, rounding_mode='floor')
        code_indices = flat_indices.remainder(codes)
        if paths.shape[-1] > 0:
            paths = paths.gather(
                1,
                parent_indices.unsqueeze(-1).expand(-1, -1, paths.shape[-1]),
            )
        else:
            paths = paths.expand(-1, keep, -1)
        paths = torch.cat([paths, code_indices.unsqueeze(-1)], dim=-1)
    return (
        paths.reshape(*leading_shape, paths.shape[-2], levels),
        path_scores.reshape(*leading_shape, path_scores.shape[-1]),
    )


def autoregressive_path_log_probabilities(
    semantic_logits,
    path_ids,
    suffix_logits=None,
    suffix_ids=None,
):
    """Evaluate hard semantic paths under the autoregressive GR distribution."""
    if semantic_logits.shape[:-1] != path_ids.shape:
        raise ValueError("path ids must match semantic logit positions")
    log_p = F.log_softmax(semantic_logits.float(), dim=-1)
    path_log_p = log_p.gather(
        dim=-1, index=path_ids.long().unsqueeze(-1)
    ).squeeze(-1).sum(dim=-1)
    if (suffix_logits is None) != (suffix_ids is None):
        raise ValueError("suffix logits and ids must be supplied together")
    if suffix_logits is not None:
        if suffix_logits.shape[:-1] != path_log_p.shape:
            raise ValueError("suffix logits must match the path batch")
        suffix_log_p = F.log_softmax(suffix_logits.float(), dim=-1)
        path_log_p = path_log_p + suffix_log_p.gather(
            dim=-1, index=suffix_ids.long().unsqueeze(-1)
        ).squeeze(-1)
    return path_log_p


def candidate_log_marginal_scores(
    semantic_logits,
    assignment_probabilities,
    suffix_logits=None,
    suffix_ids=None,
):
    """Score candidate items by a per-level latent SID marginal.

    ``semantic_logits`` and ``assignment_probabilities`` have shape
    ``[..., levels, codes]``.  The score is the soft-prefix approximation

        sum_l log sum_k q_l(k | item) p_l(k | history, candidate prefix).

    A hard collision suffix can be included in the same item score.
    """
    if semantic_logits.shape != assignment_probabilities.shape:
        raise ValueError(
            "semantic_logits and assignment_probabilities must have equal shape"
        )
    if semantic_logits.dim() < 3:
        raise ValueError("candidate tensors must end in [levels, codes]")
    if not semantic_logits.is_floating_point():
        raise TypeError("semantic_logits must be floating point")
    if not assignment_probabilities.is_floating_point():
        raise TypeError("assignment_probabilities must be floating point")
    if not torch.isfinite(semantic_logits).all():
        raise ValueError("semantic_logits must be finite")
    if not torch.isfinite(assignment_probabilities).all():
        raise ValueError("assignment_probabilities must be finite")
    if (assignment_probabilities < 0).any():
        raise ValueError("assignment probabilities must be non-negative")

    tiny = torch.finfo(assignment_probabilities.dtype).tiny
    normalized_q = assignment_probabilities / assignment_probabilities.sum(
        dim=-1, keepdim=True
    ).clamp_min(tiny)
    log_q = normalized_q.clamp_min(tiny).log()
    log_p = F.log_softmax(semantic_logits.float(), dim=-1).to(log_q.dtype)
    scores = torch.logsumexp(log_q + log_p, dim=-1).sum(dim=-1)

    if (suffix_logits is None) != (suffix_ids is None):
        raise ValueError("suffix_logits and suffix_ids must be supplied together")
    if suffix_logits is not None:
        if suffix_logits.shape[:-1] != scores.shape:
            raise ValueError("suffix logits must match candidate leading dimensions")
        if suffix_ids.shape != scores.shape:
            raise ValueError("suffix ids must match candidate score dimensions")
        if suffix_ids.min() < 0 or suffix_ids.max() >= suffix_logits.shape[-1]:
            raise ValueError("suffix ids are out of range")
        suffix_log_p = F.log_softmax(suffix_logits.float(), dim=-1)
        suffix_score = suffix_log_p.gather(
            dim=-1, index=suffix_ids.long().unsqueeze(-1)
        ).squeeze(-1)
        scores = scores + suffix_score.to(scores.dtype)
    return scores


def normalized_latent_marginal_loss(candidate_scores, positive_index=0):
    """Normalize latent overlap scores across positive and negative items."""
    if candidate_scores.dim() != 2:
        raise ValueError("candidate_scores must have shape [batch, candidates]")
    if candidate_scores.shape[1] < 2:
        raise ValueError("at least one negative item is required")
    if not 0 <= int(positive_index) < candidate_scores.shape[1]:
        raise ValueError("positive_index is outside the candidate dimension")
    labels = torch.full(
        (candidate_scores.shape[0],),
        int(positive_index),
        dtype=torch.long,
        device=candidate_scores.device,
    )
    return F.cross_entropy(candidate_scores.float(), labels)


def in_batch_candidate_indices(target_item_ids, negative_count):
    """Select deterministic, distinct in-batch target negatives per history."""
    item_ids = target_item_ids.reshape(-1)
    if item_ids.numel() < 2:
        raise ValueError("in-batch negatives require at least two examples")
    if negative_count < 1:
        raise ValueError("negative_count must be positive")
    batch_size = item_ids.numel()
    rows = torch.arange(batch_size, device=item_ids.device)
    offsets = torch.arange(1, batch_size, device=item_ids.device)
    ordered = (rows.unsqueeze(1) + offsets.unsqueeze(0)) % batch_size
    valid = item_ids[ordered] != item_ids.unsqueeze(1)
    available = valid.sum(dim=1)
    effective_count = min(int(negative_count), int(available.min().item()))
    if effective_count < 1:
        raise ValueError("batch contains no distinct target item for a negative")
    negatives = torch.stack(
        [ordered[row][valid[row]][:effective_count] for row in range(batch_size)]
    )
    return torch.cat([rows.unsqueeze(1), negatives], dim=1)
