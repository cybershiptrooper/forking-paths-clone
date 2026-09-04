"""Loss functions for circuit discovery optimization.

Local objectives (Objective 3 / contrastive KL):
    Take (clean_logits, masked_logits, position_mask) and return a
    differentiable scalar loss. Per-token, per-chain.

Global objectives (Objectives 1 & 2 / faithfulness, reward):
    Take (chain_logprobs_masked, chain_logprobs_clean, answer_ids, num_answers)
    and return a differentiable scalar loss. Operate across all chains via
    importance sampling.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional

from utils.importance_sampling import importance_weights, snis_answer_probs


# Per-call diagnostics side-channel.  Loss functions overwrite this dict on
# every call with saturation statistics (effective sample size, softmax
# entropy, pair-saturation fraction, ...); the SNP trainer copies it into
# training_metrics.jsonl at log time.  Values must be plain Python floats.
LAST_DIAGNOSTICS: dict = {}


def _set_diagnostics(**kwargs) -> None:
    LAST_DIAGNOSTICS.clear()
    LAST_DIAGNOSTICS.update(kwargs)


def kl_divergence_loss(
    clean_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """KL(softmax(clean) || softmax(masked)), averaged over valid positions and batch.

    Args:
        clean_logits: (batch, seq_len, vocab) — detached reference logits
        masked_logits: (batch, seq_len, vocab) — logits from masked model (has grad)
        position_mask: (batch, seq_len) — 1 for tokens to include, 0 to ignore.
            Typically 1 for continuation tokens, 0 for prompt prefix.

    Returns:
        Scalar KL divergence loss (differentiable w.r.t. masked_logits).
    """
    seq_len = clean_logits.shape[1]
    KL_CHUNK = 4096  # chunk along seq dim to avoid materialising (seq, vocab) all at once

    if seq_len <= KL_CHUNK:
        # Short sequence — original path (no chunking overhead)
        clean_log_probs = F.log_softmax(clean_logits.detach().float(), dim=-1)
        masked_log_probs = F.log_softmax(masked_logits.float(), dim=-1)
        kl = F.kl_div(
            masked_log_probs, clean_log_probs, log_target=True, reduction="none"
        ).sum(dim=-1)
        if position_mask is not None:
            return (kl * position_mask).sum() / position_mask.sum().clamp(min=1)
        return kl.mean()

    # Long sequence — chunk to cap peak memory at ~512 × vocab × 4 bytes × 3 tensors
    kl_sum = torch.tensor(0.0, device=masked_logits.device)
    mask_sum = torch.tensor(0.0, device=masked_logits.device) if position_mask is not None else None
    count = 0
    for s in range(0, seq_len, KL_CHUNK):
        e = min(s + KL_CHUNK, seq_len)
        c_lp = F.log_softmax(clean_logits[:, s:e].detach().float(), dim=-1)
        m_lp = F.log_softmax(masked_logits[:, s:e].float(), dim=-1)
        kl_chunk = F.kl_div(m_lp, c_lp, log_target=True, reduction="none").sum(dim=-1)
        if position_mask is not None:
            pm = position_mask[:, s:e]
            kl_sum = kl_sum + (kl_chunk * pm).sum()
            mask_sum = mask_sum + pm.sum()
        else:
            kl_sum = kl_sum + kl_chunk.sum()
            count += kl_chunk.numel()
    if position_mask is not None:
        return kl_sum / mask_sum.clamp(min=1)
    return kl_sum / count


def log_prob_loss(
    clean_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
    token_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Negative log-probability of actual tokens under the masked model.

    Maximising log-prob (minimising this loss) encourages the masked model
    to assign high probability to the same tokens as the original sequence.

    Args:
        clean_logits: Unused (kept for interface compatibility).
        masked_logits: (batch, seq_len, vocab) — logits from masked model.
        position_mask: (batch, seq_len) — 1 for tokens to include.
        token_ids: (batch, seq_len) — the actual input token IDs.
            Required for this objective.

    Returns:
        Scalar negative mean log-prob (differentiable w.r.t. masked_logits).
    """
    if token_ids is None:
        raise ValueError("log_prob_loss requires token_ids argument")

    log_probs = F.log_softmax(masked_logits.float(), dim=-1)
    # logits at position i predict token i+1
    targets = token_ids[:, 1:]  # (batch, seq_len - 1)
    token_lp = log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    if position_mask is not None:
        pm = position_mask[:, :-1]
        return -(token_lp * pm).sum() / pm.sum().clamp(min=1)
    return -token_lp.mean()


def answer_probe_kl_loss(
    clean_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
    answer_token_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """KL(P_clean || P_masked) over a fixed set of answer tokens.

    Restricted to the rows of *masked_logits* / *clean_logits* selected by
    *position_mask* (a single 1.0 for the deterministic answer probe), and
    further restricted to the columns indexed by *answer_token_ids*.
    Both distributions are softmax-renormalised over only those columns,
    so the KL is a probability comparison on the {A, B, C, D, ...} simplex.

    Args:
        clean_logits: ``(batch, seq_len, vocab)`` — detached reference
            logits from the unablated model.
        masked_logits: ``(batch, seq_len, vocab)`` — logits from the
            masked model.
        position_mask: ``(batch, seq_len)`` — 1 at the answer-prediction
            position, 0 elsewhere.  Must be supplied (we don't average
            over the whole sequence).
        answer_token_ids: ``(num_answers,)`` token IDs to compare over.
            Bound via ``functools.partial`` at the call site.

    Returns:
        Scalar KL averaged over selected positions (typically a single
        position, so just one KL value).
    """
    if answer_token_ids is None:
        raise ValueError("answer_probe_kl_loss requires answer_token_ids")
    if position_mask is None:
        raise ValueError("answer_probe_kl_loss requires position_mask")

    ans_ids = answer_token_ids.to(masked_logits.device)
    pm = position_mask.float()
    # Select rows: the answer-prediction positions across the batch
    # (typically one row for our use case).
    sel = pm > 0  # (batch, seq_len) bool
    clean_rows = clean_logits[sel].float().detach()    # (n_sel, vocab)
    masked_rows = masked_logits[sel].float()           # (n_sel, vocab)
    if clean_rows.shape[0] == 0:
        return masked_logits.sum() * 0.0

    clean_ans = clean_rows[:, ans_ids]
    masked_ans = masked_rows[:, ans_ids]
    log_p_clean = F.log_softmax(clean_ans, dim=-1).detach()
    log_p_masked = F.log_softmax(masked_ans, dim=-1)
    p_clean = log_p_clean.exp()
    kl = (p_clean * (log_p_clean - log_p_masked)).sum(dim=-1)
    return kl.mean()


def answer_probe_target_kl_loss(
    clean_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
    answer_token_ids: Optional[torch.Tensor] = None,
    target_probs: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """KL(target || P_masked) against a fixed target distribution over the
    answer-token simplex.

    Like :func:`answer_probe_kl_loss` but the reference distribution is a
    fixed vector supplied at config time (e.g. the empirical answer
    distribution of the model on a control prompt) instead of the clean
    model's probe distribution on the same prompt.  *clean_logits* is
    unused (kept for interface compatibility).
    """
    if answer_token_ids is None:
        raise ValueError("answer_probe_target_kl_loss requires answer_token_ids")
    if position_mask is None:
        raise ValueError("answer_probe_target_kl_loss requires position_mask")
    if target_probs is None:
        raise ValueError("answer_probe_target_kl_loss requires target_probs")

    ans_ids = answer_token_ids.to(masked_logits.device)
    pm = position_mask.float()
    sel = pm > 0
    masked_rows = masked_logits[sel].float()
    if masked_rows.shape[0] == 0:
        return masked_logits.sum() * 0.0

    tp = target_probs.to(masked_logits.device).float()
    tp = (tp.clamp_min(1e-8) / tp.clamp_min(1e-8).sum()).detach()
    log_p_masked = F.log_softmax(masked_rows[:, ans_ids], dim=-1)
    kl = (tp * (tp.log() - log_p_masked)).sum(dim=-1)
    return kl.mean()


def answer_probe_reward_gap_loss(
    clean_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
    answer_token_ids: Optional[torch.Tensor] = None,
    target_answer: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Negative reward gap over the fixed answer-token simplex.

    Returns ``-(P_masked(target) - max_{a != target} P_masked(a))`` averaged
    over the selected answer-prediction positions.  Minimising this loss
    increases the masked model's confidence on *target_answer* relative
    to its strongest competitor.

    *clean_logits* is unused (kept for interface compatibility).
    """
    if answer_token_ids is None:
        raise ValueError("answer_probe_reward_gap_loss requires answer_token_ids")
    if position_mask is None:
        raise ValueError("answer_probe_reward_gap_loss requires position_mask")

    ans_ids = answer_token_ids.to(masked_logits.device)
    num_answers = ans_ids.shape[-1]
    if not (0 <= target_answer < num_answers):
        raise ValueError(
            f"target_answer={target_answer} out of range for "
            f"{num_answers} answers"
        )

    pm = position_mask.float()
    sel = pm > 0
    masked_rows = masked_logits[sel].float()
    if masked_rows.shape[0] == 0:
        return masked_logits.sum() * 0.0

    masked_ans = masked_rows[:, ans_ids]
    p = F.softmax(masked_ans, dim=-1)
    p_target = p[:, target_answer]

    other = torch.ones(num_answers, dtype=torch.bool, device=p.device)
    other[target_answer] = False
    p_best_other = p[:, other].max(dim=-1).values

    return -(p_target - p_best_other).mean()


def answer_probe_logit_margin_loss(
    clean_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
    answer_token_ids: Optional[torch.Tensor] = None,
    target_answer: int = 0,
    reduce: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Negative logit-margin between target and other answer tokens.

    Operates directly on raw logits (no softmax).  Returns
    ``-(logit[target] - reduce(logits[others]))`` averaged over selected
    positions.  Minimising this loss makes the masked model assign a
    larger raw-logit gap to the target letter.

    Args:
        clean_logits: Unused (kept for interface compatibility).
        masked_logits: ``(batch, seq_len, vocab)``.
        position_mask: ``(batch, seq_len)`` — typically a single 1.0 at
            the answer-prediction position.
        answer_token_ids: ``(num_answers,)`` token IDs.  Bound via
            ``functools.partial`` at the call site.
        target_answer: Index into ``answer_token_ids`` of the letter to
            promote.
        reduce: ``"mean"`` or ``"max"`` — how to reduce the other
            letters' logits before subtracting.
    """
    if answer_token_ids is None:
        raise ValueError("answer_probe_logit_margin_loss requires answer_token_ids")
    if position_mask is None:
        raise ValueError("answer_probe_logit_margin_loss requires position_mask")
    if reduce not in ("mean", "max"):
        raise ValueError(f"reduce must be 'mean' or 'max', got {reduce!r}")

    ans_ids = answer_token_ids.to(masked_logits.device)
    num_answers = ans_ids.shape[-1]
    if not (0 <= target_answer < num_answers):
        raise ValueError(
            f"target_answer={target_answer} out of range for "
            f"{num_answers} answers"
        )

    pm = position_mask.float()
    sel = pm > 0
    masked_rows = masked_logits[sel].float()
    if masked_rows.shape[0] == 0:
        return masked_logits.sum() * 0.0

    masked_ans = masked_rows[:, ans_ids]                  # (n_sel, num_ans)
    target_logit = masked_ans[:, target_answer]           # (n_sel,)
    other = torch.ones(num_answers, dtype=torch.bool, device=masked_ans.device)
    other[target_answer] = False
    others_logits = masked_ans[:, other]                  # (n_sel, num_ans-1)
    if reduce == "mean":
        other_red = others_logits.mean(dim=-1)
    else:
        other_red = others_logits.max(dim=-1).values
    margin = target_logit - other_red                     # (n_sel,)
    return -margin.mean()


def answer_probe_prefix_kl_loss(
    clean_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
    sentences=None,
    prefix_len: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """Per-sentence-then-mean prefix-KL loss (Bogdan / thought-anchors style).

    Mirrors what :func:`expts.thought_anchor_analysis.compute_suppression_scores`
    measures per ablation: at each prefix sentence *s*, the per-token KL
    between clean and masked next-token distributions averaged across the
    tokens of *s*; then the mean across sentences. Each prefix sentence
    contributes equally regardless of length.

    The KL is over the full vocab (no answer-token simplex
    restriction) — that's the metric TA's per-sentence KL is built on.
    The objective ignores ``position_mask`` (it iterates over ``sentences``
    directly); accepted for interface compatibility with the SNP /
    attribution training loops.

    Args:
        clean_logits: ``(batch=1, full_seq_len, vocab)``.
        masked_logits: ``(batch=1, full_seq_len, vocab)``.
        sentences: bound via ``functools.partial`` from the call site —
            iterable of objects with ``.start`` / ``.end`` token indices
            (inclusive end). Only sentences whose token range falls in
            the prefix region (``end < prefix_len``) contribute.
        prefix_len: bound via partial — number of prefix tokens in the
            sequence. Used to restrict KL to prefix-internal predictions.

    Returns:
        Scalar mean KL across prefix sentences. Higher = larger
        per-sentence drift from clean.
    """
    if sentences is None or prefix_len is None:
        raise ValueError(
            "answer_probe_prefix_kl_loss requires `sentences` and `prefix_len` "
            "to be bound via functools.partial."
        )
    # KL(P_clean || P_masked) per token, full vocab, fp32.
    log_p_clean = F.log_softmax(clean_logits.float().detach(), dim=-1)  # (1, T, V)
    log_p_masked = F.log_softmax(masked_logits.float(), dim=-1)         # (1, T, V)
    p_clean = log_p_clean.exp()
    kl_per_token = (p_clean * (log_p_clean - log_p_masked)).sum(dim=-1)  # (1, T)
    kl_per_token = kl_per_token[0]  # (T,)

    # Logits at position t predict token t+1, so the KL relevant to
    # "predicting the next token of sentence s" sits at positions
    # [s.start - 1, s.end - 1] in logit space (inclusive). Token s.start
    # is predicted by logits[s.start - 1]; for s.start == 0, skip
    # (there's no prior position to predict it from).
    per_sentence = []
    for sent in sentences:
        # Only consider sentences fully inside the prefix region.
        if sent.end >= prefix_len:
            continue
        lo = max(0, sent.start - 1)
        hi = max(lo, sent.end - 1)  # inclusive
        if hi < lo:
            continue
        per_sentence.append(kl_per_token[lo:hi + 1].mean())

    if not per_sentence:
        # Defensive: no eligible prefix sentence — return 0 with grad.
        return masked_logits.sum() * 0.0
    return torch.stack(per_sentence).mean()


OBJECTIVES = {
    "kl_divergence": kl_divergence_loss,
    "log_prob": log_prob_loss,
    "answer_probe_kl": answer_probe_kl_loss,
    "answer_probe_reward_gap": answer_probe_reward_gap_loss,
    "answer_probe_logit_margin": answer_probe_logit_margin_loss,
    "answer_probe_prefix_kl": answer_probe_prefix_kl_loss,
    "answer_probe_target_kl": answer_probe_target_kl_loss,
}


# ---------------------------------------------------------------------------
# Global objectives (Objectives 1 & 2)
# ---------------------------------------------------------------------------


def answer_distribution_kl_loss(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    chain_lengths: Optional[torch.Tensor] = None,
    is_method: str = "snis",
    is_temperature: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    """KL(P_clean || P_m) over the answer distribution — Objective 1 (Faithfulness).

    P_clean is estimated by counting (chains were sampled from clean model).
    P_m is estimated via self-normalized importance sampling.

    Args:
        chain_logprobs_masked: (N,) log-probs under masked model (has grad)
        chain_logprobs_clean: (N,) log-probs under clean model (detached)
        answer_ids: (N,) integer answer IDs
        num_answers: number of distinct answers
        chain_lengths: (N,) per-chain continuation lengths (required for
            is_method='geometric_mean').
        is_method: importance-sampling method, forwarded to importance_weights.
        is_temperature: scalar T for is_method='tempered_snis'.

    Returns:
        Scalar KL divergence (differentiable w.r.t. chain_logprobs_masked).
    """
    N = chain_logprobs_masked.shape[0]
    device = chain_logprobs_masked.device

    # P_clean: simple counting (chains were sampled from the clean model)
    p_clean = torch.zeros(num_answers, device=device)
    for a in range(num_answers):
        p_clean[a] = (answer_ids == a).float().sum() / N

    # P_m: importance sampling
    w = importance_weights(
        chain_logprobs_masked, chain_logprobs_clean,
        method=is_method, chain_lengths=chain_lengths,
        temperature=is_temperature,
    )
    p_m = snis_answer_probs(w, answer_ids, num_answers)

    # KL(P_clean || P_m) — only over answers with non-zero P_clean
    active = p_clean > 0
    kl = (
        p_clean[active]
        * torch.log(p_clean[active] / p_m[active].clamp(min=1e-10))
    ).sum()
    return kl


def reward_gap_loss(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    target_answer: int = 0,
    chain_lengths: Optional[torch.Tensor] = None,
    is_method: str = "snis",
    is_temperature: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    """Negative reward gap -(P_m(A) - max_{a!=A} P_m(a)) — Objective 2 (Reward).

    Minimizing this loss maximizes the probability gap for the target answer.

    Args:
        chain_logprobs_masked: (N,) log-probs under masked model (has grad)
        chain_logprobs_clean: (N,) log-probs under clean model (detached)
        answer_ids: (N,) integer answer IDs
        num_answers: number of distinct answers
        target_answer: which answer ID to promote (default 0)
        chain_lengths: (N,) per-chain continuation lengths (required for
            is_method='geometric_mean').
        is_method: importance-sampling method, forwarded to importance_weights.
        is_temperature: scalar T for is_method='tempered_snis'.

    Returns:
        Scalar loss (differentiable w.r.t. chain_logprobs_masked).
    """
    w = importance_weights(
        chain_logprobs_masked, chain_logprobs_clean,
        method=is_method, chain_lengths=chain_lengths,
        temperature=is_temperature,
    )
    p_m = snis_answer_probs(w, answer_ids, num_answers)

    p_target = p_m[target_answer]
    other_mask = torch.ones(num_answers, dtype=torch.bool, device=p_m.device)
    other_mask[target_answer] = False

    if other_mask.any():
        p_best_other = p_m[other_mask].max()
    else:
        p_best_other = torch.zeros(1, device=p_m.device)

    return -(p_target - p_best_other)


def answer_distribution_kl_loss_weighted(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    chain_lengths: Optional[torch.Tensor] = None,
    is_method: str = "snis",
    is_temperature: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    """KL(P_clean || P_m) — Forking Paths Eq. 1, applied symmetrically.

    Both P_clean and P_m are within-sample weighted histograms (Eq. 1 of
    Bigelow et al. 2024). Each chain is weighted by its sample probability
    under the respective model:

        P_clean(a) = sum_{s: a_s=a} softmax(chain_logprobs_clean)[s]
        P_m(a)     = sum_{s: a_s=a} softmax(chain_logprobs_masked)[s]

    Neither side is an unbiased estimator of a true marginal — they are
    the paper's "outcome distribution" objects, computed under each model.
    Gradient flows through chain_logprobs_masked via P_m.

    Args:
        chain_logprobs_masked: (N,) log-probs under masked model (has grad)
        chain_logprobs_clean: (N,) log-probs under clean model (detached)
        answer_ids: (N,) integer answer IDs
        num_answers: number of distinct answers

    Returns:
        Scalar KL divergence (differentiable w.r.t. chain_logprobs_masked).
    """
    device = chain_logprobs_masked.device

    # P_clean: weighted by sample probability under clean model (detached)
    sample_weights_clean = torch.softmax(chain_logprobs_clean.detach(), dim=0)
    p_clean = torch.zeros(num_answers, device=device)
    for a in range(num_answers):
        mask = (answer_ids == a).float().to(device)
        p_clean[a] = (sample_weights_clean * mask).sum()

    # P_m: weighted by sample probability under masked model (has grad)
    sample_weights_m = torch.softmax(chain_logprobs_masked, dim=0)
    p_m = torch.zeros(num_answers, device=device)
    for a in range(num_answers):
        mask = (answer_ids == a).float().to(device)
        p_m[a] = (sample_weights_m * mask).sum()

    # KL(P_clean || P_m) — only over answers with non-zero P_clean
    active = p_clean > 0
    kl = (
        p_clean[active]
        * torch.log(p_clean[active] / p_m[active].clamp(min=1e-10))
    ).sum()
    return kl


# ---------------------------------------------------------------------------
# Candidate-set objectives (open-ended answers)
# ---------------------------------------------------------------------------
#
# For open-ended datasets (e.g. MATH) the single-token answer probe does not
# apply: answers are multi-token strings.  These objectives operate on a
# fixed **candidate answer bank** — a small set of complete answer strings
# sampled once from the clean model at the probe context and graded once
# (see expts/direct_answer_circuit_discovery/build_answer_bank.py).  Each
# continuation passed to the discovery algorithm is (probe suffix + one
# candidate's answer tokens); ``chain_logprobs_*`` are the summed
# log-probabilities of that continuation.  The probe-suffix tokens are
# identical across candidates, so their contribution cancels in the softmax
# (candidate_reward_gap), in the margin (candidate_logprob_margin), and in
# the self-normalized weights (candidate_snis_reward_gap).
#
# ``answer_ids`` maps each candidate to an answer cluster (equivalent
# formats of the same answer share a cluster); ``target_answer`` is the
# gold cluster.


def candidate_reward_gap_loss(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    target_answer: int = 0,
    **kwargs,
) -> torch.Tensor:
    """-(P(gold) - max P(wrong)) with P = softmax over the candidate set.

    The direct generalization of ``answer_probe_reward_gap_loss``: the
    {A, B, C, D} letter simplex is replaced by the candidate answer bank,
    and single-token logits by teacher-forced sequence log-probabilities.
    No importance sampling; ``chain_logprobs_clean`` is unused.

    Args:
        chain_logprobs_masked: (N,) sequence log-probs under the masked
            model, one per candidate (has grad).
        chain_logprobs_clean: (N,) unused; kept for the global-objective
            call signature.
        answer_ids: (N,) cluster id per candidate.
        num_answers: number of clusters.
        target_answer: gold cluster id.

    Returns:
        Scalar loss (differentiable w.r.t. chain_logprobs_masked).
    """
    device = chain_logprobs_masked.device
    ids = answer_ids.to(device)
    p = torch.softmax(chain_logprobs_masked, dim=0)
    p_cluster = torch.zeros(num_answers, device=device).index_add(0, ids, p)

    p_target = p_cluster[target_answer]
    other = torch.ones(num_answers, dtype=torch.bool, device=device)
    other[target_answer] = False
    p_best_other = (
        p_cluster[other].max() if other.any()
        else torch.zeros((), device=device)
    )
    with torch.no_grad():
        pd = p.detach()
        entropy = float(-(pd * (pd + 1e-12).log()).sum().item())
        _set_diagnostics(
            candidate_softmax_entropy=entropy,
            candidate_softmax_max_prob=float(pd.max().item()),
            p_target_cluster=float(p_target.detach().item()),
        )
    return -(p_target - p_best_other)


def candidate_logprob_margin_loss(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    target_answer: int = 0,
    **kwargs,
) -> torch.Tensor:
    """-(logprob(gold) - max logprob(wrong)) over the candidate set.

    Generalization of ``answer_probe_logit_margin_loss``: raw sequence
    log-probabilities, no softmax normalization.  A cluster's log-prob is
    the logsumexp over its member candidates.  Unbounded, unlike the
    probability-gap form.
    """
    device = chain_logprobs_masked.device
    ids = answer_ids.to(device)
    cluster_lp = torch.stack([
        torch.logsumexp(chain_logprobs_masked[ids == a], dim=0)
        if (ids == a).any()
        else torch.tensor(float("-inf"), device=device)
        for a in range(num_answers)
    ])
    lp_target = cluster_lp[target_answer]
    other = torch.ones(num_answers, dtype=torch.bool, device=device)
    other[target_answer] = False
    lp_best_other = (
        cluster_lp[other].max() if other.any()
        else torch.zeros((), device=device)
    )
    return -(lp_target - lp_best_other)


def candidate_snis_reward_gap_loss(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    target_answer: int = 0,
    sample_counts: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Reward gap with self-normalized importance sampling over the bank.

    The bank candidates are treated as samples from the clean model with
    multiplicity ``sample_counts``; weights
    w_i ∝ n_i · exp(logprob_masked_i − logprob_clean_i) estimate the
    masked model's answer-cluster probabilities over the sampled support.
    Candidates that were force-included but never sampled (count 0, e.g.
    the gold answer when the clean model missed it) are excluded from the
    estimator — they are not samples.

    Args:
        sample_counts: (N,) sample multiplicities.  Required.
    """
    if sample_counts is None:
        raise ValueError("candidate_snis_reward_gap_loss requires sample_counts")
    device = chain_logprobs_masked.device
    ids = answer_ids.to(device)
    counts = sample_counts.to(device).float()

    valid = counts > 0
    log_w = (
        chain_logprobs_masked[valid]
        - chain_logprobs_clean[valid].detach()
        + counts[valid].log()
    )
    w = torch.softmax(log_w, dim=0)
    with torch.no_grad():
        wd = w.detach()
        _set_diagnostics(
            snis_ess=float(1.0 / (wd ** 2).sum().item()),
            snis_max_weight=float(wd.max().item()),
            snis_log_weight_spread=float(
                (log_w.detach().max() - log_w.detach().min()).item()
            ),
        )
    p_m = torch.zeros(num_answers, device=device).index_add(0, ids[valid], w)

    p_target = p_m[target_answer]
    other = torch.ones(num_answers, dtype=torch.bool, device=device)
    other[target_answer] = False
    p_best_other = (
        p_m[other].max() if other.any()
        else torch.zeros((), device=device)
    )
    return -(p_target - p_best_other)


def candidate_pairwise_logistic_loss(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    target_answer: int = 0,
    sample_counts: Optional[torch.Tensor] = None,
    chain_lengths: Optional[torch.Tensor] = None,
    beta: float = 10.0,
    **kwargs,
) -> torch.Tensor:
    """Pairwise logistic ranking over (target, non-target) candidate pairs.

    For each pair (i in the target cluster, j outside it), the margin is
    the difference of *per-token mean* log-probabilities, with the clean
    model's margin subtracted as a baseline:

        x_ij = (lp_m[i] - lp_c[i]) / len_i  -  (lp_m[j] - lp_c[j]) / len_j

    and the loss is  mean_ij -log sigmoid(beta * x_ij).

    Design notes (vs the softmax / SNIS candidate objectives):
    - Per-token mean normalization removes the sequence-length term that
      dominates raw sequence log-probs (a 512-token candidate sits
      hundreds of nats below a 150-token one for length reasons alone).
    - The clean-model baseline cancels content confounds (terminated vs
      mid-reasoning tokens have different per-token entropy), so at
      initialization (mask ~ identity) every margin is ~0 — the sigmoid
      starts at maximal gradient by construction.
    - There is no normalization across the candidate set, so no single
      argmax candidate can absorb all probability mass; the loss keeps a
      nonzero gradient until *every* pair is confidently ordered.

    Args:
        chain_lengths: (N,) token count per candidate.  Required.
        beta: sigmoid temperature on the per-token-mean margin.
    """
    if chain_lengths is None:
        raise ValueError("candidate_pairwise_logistic_loss requires chain_lengths")
    device = chain_logprobs_masked.device
    ids = answer_ids.to(device)
    lens = chain_lengths.to(device).float()
    if sample_counts is not None:
        valid = sample_counts.to(device) > 0
    else:
        valid = torch.ones_like(ids, dtype=torch.bool)

    per_tok = (chain_logprobs_masked - chain_logprobs_clean.detach()) / lens
    tgt = valid & (ids == target_answer)
    non = valid & (ids != target_answer)
    if not tgt.any() or not non.any():
        raise ValueError(
            "candidate_pairwise_logistic_loss needs at least one candidate "
            "in and one outside the target cluster."
        )
    margins = per_tok[tgt].unsqueeze(1) - per_tok[non].unsqueeze(0)  # (T, O)
    loss = F.softplus(-beta * margins).mean()
    with torch.no_grad():
        md = margins.detach()
        _set_diagnostics(
            pairwise_frac_saturated=float(
                ((beta * md).abs() > 4.0).float().mean().item()
            ),
            pairwise_mean_margin=float(md.mean().item()),
            pairwise_margin_std=float(md.std().item()) if md.numel() > 1 else 0.0,
        )
    return loss


def candidate_pairwise_logistic_length_loss(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    target_answer: int = 0,
    sample_counts: Optional[torch.Tensor] = None,
    chain_lengths: Optional[torch.Tensor] = None,
    beta: float = 10.0,
    **kwargs,
) -> torch.Tensor:
    """Pairwise logistic ranking with added shorter-preferred length pairs.

    Same construction as ``candidate_pairwise_logistic_loss`` — margins are
    differences of per-token mean log-probabilities with the clean model's
    margin subtracted — but the pair set is extended: in addition to every
    (target-cluster candidate, other candidate) pair, every ordered pair of
    *target-cluster* candidates (shorter one preferred) contributes a
    logistic term.  The added pairs make the loss favor shorter
    terminated-with-the-trace's-answer continuations over longer ones —
    a rank-based length preference with no token-count threshold anywhere
    in the objective.  All pairs are pooled with equal weight.
    """
    if chain_lengths is None:
        raise ValueError(
            "candidate_pairwise_logistic_length_loss requires chain_lengths"
        )
    device = chain_logprobs_masked.device
    ids = answer_ids.to(device)
    lens = chain_lengths.to(device).float()
    if sample_counts is not None:
        valid = sample_counts.to(device) > 0
    else:
        valid = torch.ones_like(ids, dtype=torch.bool)

    per_tok = (chain_logprobs_masked - chain_logprobs_clean.detach()) / lens
    tgt = valid & (ids == target_answer)
    non = valid & (ids != target_answer)
    if not tgt.any() or not non.any():
        raise ValueError(
            "candidate_pairwise_logistic_length_loss needs at least one "
            "candidate in and one outside the target cluster."
        )
    margins = (
        per_tok[tgt].unsqueeze(1) - per_tok[non].unsqueeze(0)
    ).flatten()                                              # target vs other
    tgt_idx = tgt.nonzero(as_tuple=True)[0]
    len_margins = []
    for a in range(len(tgt_idx)):
        for b in range(len(tgt_idx)):
            i, j = tgt_idx[a], tgt_idx[b]
            if lens[i] < lens[j]:                            # i shorter: prefer i
                len_margins.append(per_tok[i] - per_tok[j])
    all_margins = (
        torch.cat([margins, torch.stack(len_margins)])
        if len_margins else margins
    )
    loss = F.softplus(-beta * all_margins).mean()
    with torch.no_grad():
        md = all_margins.detach()
        _set_diagnostics(
            pairwise_frac_saturated=float(
                ((beta * md).abs() > 4.0).float().mean().item()
            ),
            pairwise_mean_margin=float(md.mean().item()),
            pairwise_margin_std=float(md.std().item()) if md.numel() > 1 else 0.0,
            n_length_pairs=float(len(len_margins)),
        )
    return loss


def candidate_target_likelihood_loss(
    chain_logprobs_masked: torch.Tensor,
    chain_logprobs_clean: torch.Tensor,
    answer_ids: torch.Tensor,
    num_answers: int,
    target_answer: int = 0,
    sample_counts: Optional[torch.Tensor] = None,
    chain_lengths: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Negative per-token mean log-likelihood of the target-cluster candidates.

    Maximum likelihood on the terminated-with-the-trace's-answer
    continuations only — the masked model is pushed to make those
    continuations likely, with the sparsity penalty doing the compression.
    Dense per-token gradient, no importance sampling, no cross-candidate
    normalization.  ``chain_logprobs_clean`` is unused.
    """
    if chain_lengths is None:
        raise ValueError("candidate_target_likelihood_loss requires chain_lengths")
    device = chain_logprobs_masked.device
    ids = answer_ids.to(device)
    lens = chain_lengths.to(device).float()
    if sample_counts is not None:
        valid = sample_counts.to(device) > 0
    else:
        valid = torch.ones_like(ids, dtype=torch.bool)
    tgt = valid & (ids == target_answer)
    if not tgt.any():
        raise ValueError(
            "candidate_target_likelihood_loss needs at least one target-cluster "
            "candidate."
        )
    per_tok = chain_logprobs_masked[tgt] / lens[tgt]
    with torch.no_grad():
        _set_diagnostics(
            target_mean_per_token_logprob=float(per_tok.detach().mean().item()),
        )
    return -per_tok.mean()


# ---------------------------------------------------------------------------
# Boundary-hazard objectives.
#
# These do not use sequence log-probabilities at all.  For each bank
# candidate, teacher-forced through the masked model, the trainer reads a
# per-boundary termination hazard (in the survival-analysis sense):
#
#     h_b = p_m(wrap-up event | prefix, candidate tokens up to boundary b)
#
# Qwen3 terminates its reasoning through a multi-token sequence (paragraph
# break -> final-answer text -> ``</think>``), so the decision points are
# the paragraph breaks, and the event read at each one is the total
# probability of the *wrap-up head token set* — the tokens observed to
# start the final-answer text on this prompt, plus ``</think>`` itself
# (see build_boundary_data.py).  Each h_b is a single-position next-token
# readout, so no product over hundreds of per-token terms ever enters the
# loss.  "Eligible" boundaries are those where the clean model's forced
# answer probe (append ``</think>`` + probe suffix at the boundary)
# already yields the trace's own answer — the per-position version of the
# bank's cluster-0 rule, which excludes the degenerate "terminate
# immediately with a wrong answer" optimum.
#
# Inputs (per flattened bank candidate c):
#     log_h[c]      (B_c,) log h_b under the masked model (has grad)
#     eligible[c]   (B_c,) bool, probe-correct boundaries
#     clean_log_h[c](B_c,) log h_b under the clean model (no grad)
#     gaps[c]       (B_c,) tokens from boundary b to the next boundary
#                   (horizon-capped for the last boundary)
#
# Each function returns a scalar averaged over candidates that have at
# least one eligible boundary.  The trainer (subclass
# ``NodewiseSubnetworkProbingBoundaryHazard``) makes ``log_h`` leaf
# tensors, calls the loss, and pushes ``leaf.grad`` back through the
# model forward — the same two-pass structure used for chain log-probs.
# ---------------------------------------------------------------------------


def _log1m_exp(log_h: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(log_h)), numerically safe for log_h < 0."""
    return torch.log1p(-torch.exp(log_h).clamp(max=1.0 - 1e-6))


def boundary_stop_prob_loss(
    log_h: List[torch.Tensor],
    eligible: List[torch.Tensor],
    clean_log_h: List[torch.Tensor],
    gaps: List[torch.Tensor],
    horizon: int,
    **kwargs,
) -> torch.Tensor:
    """-mean_c log P(stop at an eligible boundary within the horizon | path c).

    Discrete first-passage decomposition along the teacher-forced path:
    P(stop at boundary b) = h_b * prod_{b' < b} (1 - h_{b'}), summed over
    eligible boundaries only — survival multiplies over *all* boundaries
    (stopping anywhere ends the chain), but only probe-correct stopping
    points are rewarded.
    """
    per_chain = []
    diag_p = []
    for lh, el in zip(log_h, eligible):
        if not el.any():
            continue
        log_surv = torch.cumsum(_log1m_exp(lh), dim=0)
        # survival *before* boundary b: shift right by one.
        log_surv_before = torch.cat(
            [log_surv.new_zeros(1), log_surv[:-1]], dim=0
        )
        log_stop = log_surv_before + lh                     # (B,)
        lp = torch.logsumexp(log_stop[el], dim=0)
        per_chain.append(lp)
        diag_p.append(float(lp.detach().exp().item()))
    if not per_chain:
        raise ValueError(
            "boundary_stop_prob_loss: no candidate has any eligible "
            "boundary. At early analysis points (answer not yet decided) "
            "this can happen legitimately — the prompt is then unusable "
            "for the binary-guard hazard objectives; consider "
            "boundary_stop_prob_soft."
        )
    loss = -torch.stack(per_chain).mean()
    _set_diagnostics(
        stop_prob_mean=float(sum(diag_p) / len(diag_p)),
        stop_prob_max=float(max(diag_p)),
        n_chains_with_eligible=float(len(diag_p)),
        frac_chains_with_eligible=float(len(diag_p) / len(log_h)),
    )
    return loss


def boundary_stop_prob_soft_loss(
    log_h: List[torch.Tensor],
    eligible: List[torch.Tensor],
    clean_log_h: List[torch.Tensor],
    gaps: List[torch.Tensor],
    horizon: int,
    probe_p_trace: Optional[List[torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Stop probability with the answer-preservation guard made continuous.

    Identical first-passage decomposition to ``boundary_stop_prob_loss``,
    but instead of restricting the rewarded sum to boundaries whose forced
    answer probe returns the trace's answer (a binary eligible/ineligible
    filter), every boundary's stopping term is weighted by the clean
    model's forced-probe probability of the trace's answer at that
    boundary:

        P(stop at a boundary AND conclude with the trace's answer)
          ~= sum_b [prod_{b'<b} (1 - h_{b'})] * h_b * p_trace(b)

    This is the "answer correctness as a continuous part of the reward"
    variant, intended for early analysis points where the binary probe
    verdict flips noisily because the answer is genuinely undecided.
    Requires ``probe_p_trace`` (stored per boundary by
    build_boundary_data.py).
    """
    if probe_p_trace is None:
        raise ValueError(
            "boundary_stop_prob_soft_loss requires probe_p_trace "
            "(rebuild boundary data with build_boundary_data.py)."
        )
    per_chain = []
    diag_p = []
    for lh, pt in zip(log_h, probe_p_trace):
        log_surv = torch.cumsum(_log1m_exp(lh), dim=0)
        log_surv_before = torch.cat(
            [log_surv.new_zeros(1), log_surv[:-1]], dim=0
        )
        log_stop = log_surv_before + lh                      # (B,)
        log_pt = pt.to(lh.device).float().clamp(min=1e-8).log()
        lp = torch.logsumexp(log_stop + log_pt, dim=0)
        per_chain.append(lp)
        diag_p.append(float(lp.detach().exp().item()))
    loss = -torch.stack(per_chain).mean()
    _set_diagnostics(
        stop_prob_mean=float(sum(diag_p) / len(diag_p)),
        stop_prob_max=float(max(diag_p)),
        n_chains_with_eligible=float(len(diag_p)),
    )
    return loss


def boundary_hazard_lift_loss(
    log_h: List[torch.Tensor],
    eligible: List[torch.Tensor],
    clean_log_h: List[torch.Tensor],
    gaps: List[torch.Tensor],
    horizon: int,
    **kwargs,
) -> torch.Tensor:
    """-mean over eligible boundaries of (log h_masked - log h_clean).

    The clean-model hazard baseline controls for boundaries where stopping
    is naturally implausible and makes the objective comparable across
    prompts.  At initialization (mask ~ identity) every term is ~0.
    """
    lifts = []
    for lh, el, clh in zip(log_h, eligible, clean_log_h):
        if not el.any():
            continue
        lifts.append((lh - clh.detach())[el])
    if not lifts:
        raise ValueError("boundary_hazard_lift_loss: no eligible boundaries.")
    n_with = sum(1 for el in eligible if bool(el.any()))
    lifts = torch.cat(lifts)
    with torch.no_grad():
        _set_diagnostics(
            hazard_lift_mean=float(lifts.detach().mean().item()),
            hazard_lift_max=float(lifts.detach().max().item()),
            n_chains_with_eligible=float(n_with),
            frac_chains_with_eligible=float(n_with / len(log_h)),
        )
    return -lifts.mean()


def boundary_expected_length_loss(
    log_h: List[torch.Tensor],
    eligible: List[torch.Tensor],
    clean_log_h: List[torch.Tensor],
    gaps: List[torch.Tensor],
    horizon: int,
    **kwargs,
) -> torch.Tensor:
    """mean_c E[tokens until stopping, capped at the horizon | path c] / horizon.

    E[min(T, H)] = sum_b S_b * gap_b with S_b the probability of surviving
    through boundary b (over all boundaries — this variant does not use the
    eligibility labels and therefore also rewards stopping at boundaries
    whose forced-probe answer is wrong; included as the pure
    remaining-length objective).
    """
    per_chain = []
    for lh, gp in zip(log_h, gaps):
        log_surv = torch.cumsum(_log1m_exp(lh), dim=0)
        e_len = (log_surv.exp() * gp.to(lh.device).float()).sum()
        per_chain.append(e_len / float(horizon))
    loss = torch.stack(per_chain).mean()
    with torch.no_grad():
        _set_diagnostics(
            expected_length_frac_of_horizon=float(loss.detach().item()),
        )
    return loss


def boundary_expected_length_eligible_loss(
    log_h: List[torch.Tensor],
    eligible: List[torch.Tensor],
    clean_log_h: List[torch.Tensor],
    gaps: List[torch.Tensor],
    horizon: int,
    positions: Optional[List[torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Expected tokens until stopping at an *eligible* boundary, / horizon.

    The eligibility-guarded version of ``boundary_expected_length_loss``:
    only boundaries whose forced answer probe yields the trace's own
    answer count as stopping opportunities — hazards at ineligible
    boundaries are ignored (treated as pass-through), so shortening is
    never rewarded at a point where the model would conclude with a
    different answer.  Per chain, with $B^+$ the eligible boundaries at
    token positions $t_b$ and $S^+_{<b}$ the probability of not having
    stopped at an earlier eligible boundary:

        E = sum_{b in B^+} S^+_{<b} h_b t_b  +  S^+_{all} * horizon,

    and the loss is mean_c E / horizon.  Requires ``positions``.
    """
    if positions is None:
        raise ValueError(
            "boundary_expected_length_eligible_loss requires positions"
        )
    per_chain = []
    for lh, el, pos in zip(log_h, eligible, positions):
        if not el.any():
            continue
        lh_e = lh[el]
        pos_e = pos.to(lh.device).float()[el]
        log_surv = torch.cumsum(_log1m_exp(lh_e), dim=0)
        log_surv_before = torch.cat(
            [log_surv.new_zeros(1), log_surv[:-1]], dim=0
        )
        p_stop = (log_surv_before + lh_e).exp()
        e_len = (p_stop * pos_e).sum() + log_surv[-1].exp() * float(horizon)
        per_chain.append(e_len / float(horizon))
    if not per_chain:
        raise ValueError(
            "boundary_expected_length_eligible_loss: no eligible boundaries."
        )
    loss = torch.stack(per_chain).mean()
    with torch.no_grad():
        _set_diagnostics(
            expected_length_frac_of_horizon=float(loss.detach().item()),
            n_chains_with_eligible=float(len(per_chain)),
        )
    return loss


GLOBAL_OBJECTIVES = {
    "answer_kl": answer_distribution_kl_loss,
    "answer_kl_weighted": answer_distribution_kl_loss_weighted,
    "reward_gap": reward_gap_loss,
    "candidate_reward_gap": candidate_reward_gap_loss,
    "candidate_logprob_margin": candidate_logprob_margin_loss,
    "candidate_snis_reward_gap": candidate_snis_reward_gap_loss,
    "candidate_pairwise_logistic": candidate_pairwise_logistic_loss,
    "candidate_pairwise_logistic_length": candidate_pairwise_logistic_length_loss,
    "candidate_target_likelihood": candidate_target_likelihood_loss,
}

# Boundary-hazard objectives are trained by the dedicated SNP subclass
# (``nodewise_subnetwork_probing_boundary_hazard``), which reads
# per-boundary ``</think>`` log-probs instead of summed chain log-probs.
# They are registered separately because their call signature differs from
# the chain-logprob global objectives.
HAZARD_OBJECTIVES = {
    "boundary_stop_prob": boundary_stop_prob_loss,
    "boundary_stop_prob_soft": boundary_stop_prob_soft_loss,
    "boundary_hazard_lift": boundary_hazard_lift_loss,
    "boundary_expected_length": boundary_expected_length_loss,
    "boundary_expected_length_eligible": boundary_expected_length_eligible_loss,
}


_GLOBAL_FUNC_NAMES = {fn.__name__ for fn in GLOBAL_OBJECTIVES.values()}


def is_global_objective(name: str) -> bool:
    """Check if an objective is a global (outcome-level, IS-based) objective.

    Accepts both the registry key (e.g. ``"answer_kl"``) and the Python
    function name (e.g. ``"answer_distribution_kl_loss"``).

    Boundary-hazard objectives count as global here: the SNP ``discover``
    path treats them like chain-level objectives (no per-position clean
    logits cache); the hazard subclass overrides the per-step routine.
    """
    return (
        name in GLOBAL_OBJECTIVES
        or name in _GLOBAL_FUNC_NAMES
        or is_hazard_objective(name)
    )


_HAZARD_FUNC_NAMES = {fn.__name__ for fn in HAZARD_OBJECTIVES.values()}


def is_hazard_objective(name: str) -> bool:
    """Check if an objective is a boundary-hazard objective (per-boundary
    ``</think>`` log-probs, trained by the dedicated SNP subclass)."""
    return name in HAZARD_OBJECTIVES or name in _HAZARD_FUNC_NAMES


def get_objective(name: str):
    """Get objective function by name (local or global)."""
    if name in OBJECTIVES:
        return OBJECTIVES[name]
    if name in GLOBAL_OBJECTIVES:
        return GLOBAL_OBJECTIVES[name]
    all_names = list(OBJECTIVES.keys()) + list(GLOBAL_OBJECTIVES.keys())
    raise ValueError(f"Unknown objective: {name}. Available: {all_names}")
