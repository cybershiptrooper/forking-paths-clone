"""Unit tests for subnetwork-probing gate initialization.

The scalar initializations follow the convention already used by
``ColumnSubnetworkProbing`` and the sentence-grading hyperparameter search
(notes/reports_individual_sentence_grading/individual_sentence_grading_followups.md):
``log_alpha_init`` is a float — closed = -3, half = 0, open = +2 — or the
string ``"random"`` for Uniform(-2, 2) per gate.

Covers (CPU only, no model):
- The float scale maps to the gate means the convention claims.
- ``"random"`` draws Uniform(-2, 2) per gate, as in ColumnSubnetworkProbing.
- Initializing from a mask reproduces the evaluator's top-k selection,
  honours the learnable-pool filter, and interpolates towards all-open as
  alpha grows (alpha = 1 is exactly all-open, alpha = 0 is exactly the
  supplied mask).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from utils.circuit_discovery.edits.nodewise_subnetwork_probing_sdpa import (
    NodewiseSubnetworkProbingSDPA,
    _HC_BETA,
    _hard_concrete_mean,
)
from utils.masks import (
    build_causal_filter,
    build_combined_filter,
    build_gap_filter,
    build_mode_filter,
)

S = 8
DEV = torch.device("cpu")


def _make_snp(**over):
    obj = NodewiseSubnetworkProbingSDPA.__new__(NodewiseSubnetworkProbingSDPA)
    obj.layers = [0]
    obj.mask_granularity = "pair"
    obj.target_sparsity = over.get("target_sparsity", 0.5)
    obj.log_alpha_init = over.get("log_alpha_init", 2.0)
    obj.log_alpha_init_mask_path = over.get("log_alpha_init_mask_path")
    obj.log_alpha_init_mask_alpha = over.get("log_alpha_init_mask_alpha", 1.0)
    obj.hc_beta_anneal = over.get("hc_beta_anneal", False)
    obj.hc_beta_start = over.get("hc_beta_start", _HC_BETA)
    return obj


def _filter():
    return build_combined_filter(
        build_gap_filter(S, 1), build_mode_filter(S, S, "prefix"),
        build_causal_filter(S), None,
    )


def _means_of(snp, combined_filter=None):
    la = snp._init_log_alpha("pair", 4, S, DEV, combined_filter=combined_filter)
    # pair granularity carries a leading broadcast dimension of 1
    assert la.shape == (1, S, S)
    return _hard_concrete_mean(la.detach(), beta=_HC_BETA)[0]


@pytest.mark.parametrize("init,expected,name", [
    (2.0, 1.0, "open"), (-3.0, 0.0, "closed"), (0.0, 0.5, "half"),
])
def test_the_float_convention_hits_its_gate_mean(init, expected, name):
    """closed = -3, half = 0, open = +2 on the log_alpha scale."""
    m = _means_of(_make_snp(log_alpha_init=init))
    assert torch.allclose(m, torch.full_like(m, expected), atol=1e-5)


def test_random_matches_the_column_snp_convention():
    """Uniform(-2, 2) per gate, independently drawn."""
    torch.manual_seed(0)
    la = _make_snp(log_alpha_init="random")._init_log_alpha(
        "pair", 4, S, DEV, combined_filter=_filter()).detach()
    assert la.shape == (1, S, S)
    assert float(la.min()) >= -2.0 and float(la.max()) <= 2.0
    assert la.std() > 0.5                    # actually varied per gate
    # spans both hard-closed and fully-open gates
    m = _hard_concrete_mean(la, beta=_HC_BETA)
    assert float(m.min()) == pytest.approx(0.0, abs=1e-5)
    assert float(m.max()) == pytest.approx(1.0, abs=1e-5)


def test_random_rejects_other_strings():
    with pytest.raises(ValueError, match="float or 'random'"):
        _means_of(_make_snp(log_alpha_init="uniform"))


def test_mean_to_log_alpha_is_monotone_and_symmetric():
    snp = _make_snp()
    ms = torch.linspace(0, 1, 11)
    la = snp._mean_to_log_alpha(ms, DEV)
    assert torch.all(la[1:] > la[:-1])       # strictly increasing
    assert la[0].item() == pytest.approx(-2.0)
    assert la[-1].item() == pytest.approx(2.0)
    assert la[5].item() == pytest.approx(0.0, abs=1e-6)   # m = 0.5 -> 0


def _write_mask(tmp_path, scores):
    p = tmp_path / "ta.json"
    json.dump({
        "mask_type": "NodeMask", "model_name": "test", "algorithm": "ta",
        "layers": [0], "sentences": [{"start": i, "end": i + 1, "text": ""}
                                     for i in range(S)],
        "objective_name": "kl", "metadata": {"score_readout": "raw_score"},
        "scores": scores.tolist(),
    }, open(p, "w"))
    return str(p)


def test_mask_mix_reproduces_the_evaluator_topk(tmp_path):
    torch.manual_seed(3)
    scores = torch.rand(S, S)
    path = _write_mask(tmp_path, scores)
    cf = _filter()
    snp = _make_snp(log_alpha_init_mask_path=path,
                    log_alpha_init_mask_alpha=0.0, target_sparsity=0.5)
    kept = snp._load_mask_topk(path, S, DEV, cf)
    valid = ~cf.bool()
    n_valid = int(valid.sum())
    assert int(kept.sum()) == round(0.5 * n_valid)
    assert not bool((kept & ~valid).any())          # never selects frozen cells
    # the kept set is exactly the highest-scoring learnable cells
    thresh = scores[valid].sort(descending=True).values[int(kept.sum()) - 1]
    assert float(scores[kept].min()) >= float(thresh)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_mask_mix_interpolates_towards_all_open(tmp_path, alpha):
    torch.manual_seed(3)
    scores = torch.rand(S, S)
    path = _write_mask(tmp_path, scores)
    cf = _filter()
    snp = _make_snp(log_alpha_init_mask_path=path,
                    log_alpha_init_mask_alpha=alpha, target_sparsity=0.5)
    m = _means_of(snp, cf)
    kept = snp._load_mask_topk(path, S, DEV, cf)
    assert torch.allclose(m[kept], torch.ones_like(m[kept]), atol=1e-5)
    off = m[~kept]
    assert torch.allclose(off, torch.full_like(off, alpha), atol=1e-5)
    if alpha == 1.0:                     # alpha = 1 is exactly all-open
        assert torch.allclose(m, torch.ones_like(m), atol=1e-5)


def test_mask_mix_requires_a_target_sparsity():
    with pytest.raises(ValueError, match="target_sparsity"):
        _means_of(_make_snp(log_alpha_init_mask_path="x.json",
                            target_sparsity=None))


def test_no_mask_path_falls_back_to_the_scalar_path():
    """A run without an init mask is unchanged from before this feature."""
    a = _make_snp()._init_log_alpha("pair", 4, S, DEV,
                                    combined_filter=_filter()).detach()
    assert torch.allclose(a, torch.full_like(a, 2.0))
