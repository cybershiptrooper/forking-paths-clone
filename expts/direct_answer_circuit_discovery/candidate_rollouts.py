"""Candidate-bank answer distributions after sampled rollouts (open-ended answers).

The answer-distribution task on MATH (paper Appendix "Open-ended answers")
reads the answer distribution over the clusters of a candidate answer bank:
the softmax over the teacher-forced sequence log-probabilities of every
candidate string, summed per cluster. On a rollout bank the context of the
candidates is ``prefix + rollout + probe suffix`` instead of the stored
five-sentence suffix.

Every candidate of a bank shares the probe suffix (``build_answer_bank``
verifies that the join leaves the suffix tokens unchanged), so the context
is common to all candidates of a rollout and only the answer tokens after
the suffix are scored. Candidates are batched with right padding: under the
causal mask, padding after a row's last real token cannot change that row's
scores.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def bank_paths(bank: dict):
    """Answer-token paths (after the probe suffix) of every tokenization
    variant of every candidate, with their cluster ids, in the order of
    ``answer_bank_utils.flatten_bank_candidates``."""
    from expts.direct_answer_circuit_discovery.answer_bank_utils import flatten_bank_candidates

    token_lists, cluster_ids, _ = flatten_bank_candidates(bank)
    suffix = list(bank["probe_suffix_token_ids"])
    paths = []
    for ids in token_lists:
        if list(ids[: len(suffix)]) != suffix:
            raise ValueError("candidate continuation does not start with the probe suffix")
        if len(ids) == len(suffix):
            raise ValueError("candidate continuation has no answer tokens")
        paths.append(list(ids[len(suffix):]))
    return suffix, paths, cluster_ids


def padded_path_batch(context: torch.Tensor, paths: List[List[int]]):
    """(b, L + maxlen) input ids, (b, maxlen) targets and (b, maxlen) validity."""
    b = len(paths)
    maxlen = max(len(p) for p in paths)
    device = context.device
    tgt = torch.zeros(b, maxlen, dtype=torch.long, device=device)
    valid = torch.zeros(b, maxlen, dtype=torch.bool, device=device)
    for i, p in enumerate(paths):
        tgt[i, : len(p)] = torch.tensor(p, device=device)
        valid[i, : len(p)] = True
    inp = torch.cat([context.expand(b, -1), tgt], dim=-1)
    return inp, tgt, valid


def path_logprobs_from_hidden(model, hidden: torch.Tensor, context_len: int,
                              tgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Summed log-probs of each row's path from the last hidden states.

    Only the rows predicting path tokens go through the LM head
    (positions ``context_len - 1 .. context_len + maxlen - 2``)."""
    maxlen = tgt.shape[1]
    h = hidden[:, context_len - 1: context_len - 1 + maxlen]
    logits = model.lm_head(h).float()
    lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (lp * valid).sum(-1)


@torch.no_grad()
def path_logprobs(model, context: torch.Tensor, paths: List[List[int]], rows_per_forward: int = 8) -> torch.Tensor:
    """(N,) summed log-probs of each path after ``context`` (1, L), with
    whatever masks are currently installed on the model. Float32 on CPU."""
    out = []
    L = context.shape[-1]
    for lo in range(0, len(paths), rows_per_forward):
        chunk = paths[lo: lo + rows_per_forward]
        inp, tgt, valid = padded_path_batch(context, chunk)
        with torch.amp.autocast("cuda"):
            hidden = model.model(inp).last_hidden_state
        out.append(path_logprobs_from_hidden(model, hidden, L, tgt, valid).cpu())
        del hidden
    return torch.cat(out)


def cluster_probs(logprobs: torch.Tensor, cluster_ids: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """Softmax over candidate paths, summed per cluster (differentiable)."""
    p = torch.softmax(logprobs, dim=-1)
    return torch.zeros(num_clusters, dtype=p.dtype, device=p.device).index_add(0, cluster_ids.to(p.device), p)


def kl(p_clean: torch.Tensor, p_masked: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    pc, pm = p_clean.clamp_min(eps), p_masked.clamp_min(eps)
    return (pc * (pc.log() - pm.log())).sum()
