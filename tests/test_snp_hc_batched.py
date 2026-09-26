"""Tests for ``nodewise_subnetwork_probing_hc_batched``.

Verifies (CPU, tiny random Qwen3):

1. The batched mask converter returns ``(K, 1, q, k)`` with row ``i``
   equal to the unbatched expansion of sample ``i``.
2. One full training step of the batched class produces the same
   ``log_alpha`` update as the sequential base class when both see the
   same K deterministic mask "samples" — i.e. the batched gradient is
   the sequential 1/K-scaled accumulated gradient. Micro-batching
   (``batch_chunk_size < K``) is exercised in the same test.
3. Head granularity raises; K = 1 delegates to the base implementation.
"""

import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("WANDB_MODE", "disabled")

import pytest
import torch

from utils.utils import Sentence
from utils.objectives import answer_probe_kl_loss
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
)
from utils.circuit_discovery.edits.nodewise_subnetwork_probing_hc_batched import (
    NodewiseSubnetworkProbingHCBatched,
    _expand_mask_to_log_additive_hc_batched,
)
from utils.circuit_discovery.edits.nodewise_attribution_sdpa import (
    _expand_mask_to_log_additive,
)


# ---------------------------------------------------------------------------
# Converter shape / value test (no model needed)
# ---------------------------------------------------------------------------


class _DummyModule:
    pass


def _make_module(mask, num_sents, seq_len, gap=1):
    m = _DummyModule()
    m._circuit_mask = mask
    token_to_sent = torch.full((seq_len,), -1, dtype=torch.long)
    per_sent = seq_len // (num_sents + 1)
    for s in range(num_sents):
        token_to_sent[s * per_sent:(s + 1) * per_sent] = s
    m._token_to_sent = token_to_sent
    idx = torch.arange(num_sents)
    m._gap_filter = (idx[:, None] - idx[None, :]).abs() < gap
    return m


def test_batched_converter_matches_per_sample_expansion():
    torch.manual_seed(0)
    K, S, L = 3, 5, 24
    masks = torch.rand(K, S, S).clamp(0.05, 1.0)
    mod = _make_module(masks, S, L)
    out = _expand_mask_to_log_additive_hc_batched(
        mod, L, L, None, torch.float32,
    )
    assert out.shape == (K, 1, L, L)
    for i in range(K):
        mod_i = _make_module(masks[i:i + 1], S, L)
        ref = _expand_mask_to_log_additive(mod_i, L, L, None, torch.float32)
        assert ref.shape == (1, 1, L, L)
        torch.testing.assert_close(out[i:i + 1].transpose(0, 1), ref)


def test_unbatched_mask_passes_through():
    torch.manual_seed(0)
    S, L = 5, 24
    mask = torch.rand(1, S, S).clamp(0.05, 1.0)
    mod = _make_module(mask, S, L)
    out = _expand_mask_to_log_additive_hc_batched(mod, L, L, None, torch.float32)
    ref = _expand_mask_to_log_additive(mod, L, L, None, torch.float32)
    torch.testing.assert_close(out, ref)  # (1, 1, L, L) either way


# ---------------------------------------------------------------------------
# End-to-end gradient equivalence on a tiny Qwen3
# ---------------------------------------------------------------------------


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _fixture_inputs(num_sents=6, tokens_per_sent=4, cont_len=5, vocab=128):
    torch.manual_seed(1)
    prefix_len = num_sents * tokens_per_sent + 3  # 3 leading non-sentence tokens
    input_ids = torch.randint(2, vocab, (1, prefix_len))
    sentences = [
        Sentence(start=3 + s * tokens_per_sent,
                 end=3 + (s + 1) * tokens_per_sent - 1)
        for s in range(num_sents)
    ]
    continuation = torch.randint(2, vocab, (1, cont_len))
    full_len = prefix_len + cont_len
    position_mask = torch.zeros(1, full_len)
    position_mask[0, -2] = 1.0  # single probed position, like the answer probe
    answer_ids = torch.tensor([11, 23, 37, 53])
    return input_ids, sentences, [continuation], position_mask, answer_ids


def _make_algo(cls, model, K, **extra):
    objective = partial(answer_probe_kl_loss, answer_token_ids=_ANSWER_IDS)
    objective.__name__ = answer_probe_kl_loss.__name__
    return cls(
        model=model,
        tokenizer=None,
        layers=list(range(model.config.num_hidden_layers)),
        objective_fn=objective,
        sentence_gap=1,
        mask_granularity="pair",
        num_training_steps=1,
        learning_rate=0.05,
        l0_lambda=100.0,
        sparsity_loss_mode="target_size_relu",
        target_sparsity=0.5,
        optimizer="adam",
        l0_lambda_schedule=False,  # constant λ so the single step has L0 too
        num_hc_samples_per_step=K,
        wandb_project=None,
        # Raw log_alpha readout: the HC-mean readout saturates at 1.0
        # around the +2.0 init, which would make the comparison vacuous.
        log_alpha_init=0.5,
        save_log_alpha=True,
        **extra,
    )


_ANSWER_IDS = torch.tensor([11, 23, 37, 53])


def _deterministic_sample_patch(algo, K, offsets):
    """Make ``_sample_masks`` deterministic and differentiable in log_alpha.

    Sequential base class: consecutive calls cycle through the K offsets,
    one (1, S, S) sample per call. Batched class: one call returns all K
    stacked as (K, S, S). Sample i is sigmoid(log_alpha + offsets[i]) in
    both, so the two classes see identical mask values.
    """
    if isinstance(algo, NodewiseSubnetworkProbingHCBatched):
        def sample(log_alpha, granularity, beta=None):
            s = torch.sigmoid(log_alpha + offsets.view(K, 1, 1))
            return {l: s for l in algo.layers}
    else:
        state = {"i": 0}
        def sample(log_alpha, granularity, beta=None):
            s = torch.sigmoid(log_alpha + offsets[state["i"] % K])
            state["i"] += 1
            return {l: s for l in algo.layers}
    algo._sample_masks = sample


def _run_one_step(algo, K, offsets):
    inputs, sentences, conts, pm, _ = _fixture_inputs()
    _deterministic_sample_patch(algo, K, offsets)
    node_mask = algo.discover(
        input_ids=inputs,
        sentences=sentences,
        continuations=conts,
        mask_mode="prefix",
        num_prefix_sentences=len(sentences),
        branch_rewards=None,
        position_mask_overrides=[pm],
        num_frozen_prompt_sentences=0,
    )
    return torch.tensor(node_mask.scores)


@pytest.mark.parametrize("chunk", [None, 2, 3])
def test_batched_step_matches_sequential(chunk):
    K = 4
    offsets = torch.tensor([-0.6, -0.2, 0.3, 0.9])
    model = _tiny_qwen3()
    seq = _make_algo(NodewiseSubnetworkProbingSDPA, model, K)
    scores_seq = _run_one_step(seq, K, offsets)

    model2 = _tiny_qwen3()  # identical weights (same seed)
    bat = _make_algo(
        NodewiseSubnetworkProbingHCBatched, model2, K,
        batch_chunk_size=chunk,
    )
    scores_bat = _run_one_step(bat, K, offsets)

    torch.testing.assert_close(scores_bat, scores_seq, atol=2e-4, rtol=1e-3)
    # The step must actually have moved log_alpha off its 0.5 init,
    # otherwise the equality above is vacuous.
    assert not torch.allclose(
        scores_seq, torch.full_like(scores_seq, 0.5), atol=1e-4,
    )


def test_head_granularity_raises():
    model = _tiny_qwen3()
    algo = _make_algo(NodewiseSubnetworkProbingHCBatched, model, 4)
    algo.mask_granularity = "head"
    la = algo._init_log_alpha("head", 4, 6, "cpu")
    with pytest.raises(NotImplementedError):
        algo._sample_masks(la, "head")


def test_k1_delegates_to_base():
    model = _tiny_qwen3()
    algo = _make_algo(NodewiseSubnetworkProbingHCBatched, model, 1)
    la = algo._init_log_alpha("pair", 4, 6, "cpu")
    masks = algo._sample_masks(la, "pair")
    assert masks[0].shape == (1, 6, 6)
