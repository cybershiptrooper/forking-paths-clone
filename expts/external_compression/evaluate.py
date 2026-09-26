"""Prefix-compression evaluation per (instance, method, M) + oracle.

For a method's per-sentence ranking: keep the top-M compress-region
sentences (original order), append the protected 5-sentence tail, force
the probe, and measure

- answer KL: KL(baseline || compressed) over the letter distribution,
  where baseline is the full-N-prefix distribution from screening;
- token KL: force the next sentence (sentence N, 0-indexed) after the
  compressed prefix and average full-vocab per-token KL(clean || compressed)
  over its token positions, clean = full-N-prefix + forced next sentence.

Also runs the single sentence oracle (their best answer-KL baseline): for
each M, evaluate every candidate sentence combined with the last M-1
compress-region sentences and report the best.

Outputs one JSON per method under
``results/external_compression/evals/<instance>/<method>.json``.

Usage (GPU):
    uv run python -m expts.external_compression.evaluate \
        --instance <id> --methods suppress,ta,attn_last,attn_next,oracle,reference
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional

import torch

from expts.external_compression.common import (
    DATA_DIR,
    K_KEEP,
    M_VALUES,
    MODEL_NAME,
    RESULTS_DIR,
    encode_forced,
    kl_from_distributions,
    letter_distribution,
    load_rollout,
)

SCORES_DIR = os.path.join(RESULTS_DIR, "scores")
EVALS_DIR = os.path.join(RESULTS_DIR, "evals")
# Keep-fractions evaluated in addition to their fixed M values, so the
# primary (ratio) analysis has directly comparable points across buckets.
FRACTIONS = [0.05, 0.1, 0.2, 0.4]


def m_list_for(num_compress: int) -> List[int]:
    ms = {M for M in M_VALUES if M <= num_compress}
    for f in FRACTIONS:
        ms.add(min(num_compress, max(1, round(f * num_compress))))
    return sorted(ms)


class InstanceEvaluator:
    def __init__(self, instance: dict, model, tokenizer):
        with open(os.path.join(DATA_DIR, "prompts_rendered.json")) as f:
            rendered = json.load(f)
        self.inst = instance
        self.model = model
        self.tokenizer = tokenizer
        self.qid = instance["question_id"]
        self.N = instance["prefix_length"]
        roll = load_rollout(self.qid)
        self.sents = roll["sentences"][: self.N]
        self.next_sent = roll["sentences"][self.N]
        self.num_compress = self.N - K_KEEP
        self.tail = self.sents[self.num_compress :]
        r = rendered[self.qid]
        self.prompt_str = r["prompt_str"]
        self.answer_ids = r["letter_token_ids"]
        self.letters = r["letters"]
        self.baseline_dist = instance["baseline_dist"]
        self.device = next(model.parameters()).device
        self._clean_next_rows: Optional[torch.Tensor] = None
        self._clean_next_ids: Optional[List[int]] = None

    # -- answer KL ---------------------------------------------------------

    def answer_dist(self, kept_compress_idx: List[int]) -> List[float]:
        kept = [self.sents[i] for i in sorted(kept_compress_idx)] + self.tail
        ids = encode_forced(self.tokenizer, self.prompt_str, kept)
        input_t = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            try:
                logits = self.model(input_t, logits_to_keep=1).logits
            except TypeError:
                logits = self.model(input_t, num_logits_to_keep=1).logits
        return letter_distribution(logits[0, -1], self.answer_ids)

    def answer_kl(self, kept_compress_idx: List[int]) -> float:
        return kl_from_distributions(
            self.baseline_dist, self.answer_dist(kept_compress_idx),
        )

    # -- token KL ----------------------------------------------------------

    def _next_sent_logrows(self, prefix_sents: List[str]):
        """Log-softmax rows predicting the forced next sentence's tokens."""
        text_prefix = self.prompt_str + " ".join(prefix_sents)
        text = text_prefix + " " + self.next_sent
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        boundary = len(text_prefix)
        first = next(
            i for i, (a, b) in enumerate(offsets) if max(b - 1, a) >= boundary
        )
        next_ids = ids[first:]
        n = len(next_ids)
        input_t = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            try:
                logits = self.model(input_t, logits_to_keep=n + 1).logits
            except TypeError:
                logits = self.model(input_t, num_logits_to_keep=n + 1).logits
        # rows predicting positions first..first+n-1 are at first-1..first+n-2,
        # i.e. the first n of the last n+1 kept rows.
        rows = logits[0, :n].float()
        return torch.log_softmax(rows, dim=-1), next_ids

    def token_kl(self, kept_compress_idx: List[int]) -> Optional[float]:
        if self._clean_next_rows is None:
            self._clean_next_rows, self._clean_next_ids = self._next_sent_logrows(
                self.sents,
            )
        kept = [self.sents[i] for i in sorted(kept_compress_idx)] + self.tail
        lm, next_ids = self._next_sent_logrows(kept)
        if next_ids != self._clean_next_ids:
            print("  WARNING: next-sentence tokenization mismatch; skipping token_kl")
            return None
        lc = self._clean_next_rows
        kl = (lc.exp() * (lc - lm)).sum(-1).mean()
        return float(kl.item())


def ranking_from_scores(scores: List[float], num_compress: int) -> List[int]:
    """Compress-region indices sorted by descending score (stable)."""
    idx = list(range(num_compress))
    # scores[0] is the prompt block; compress sentence i has score index i+1.
    idx.sort(key=lambda i: (-scores[i + 1], i))
    return idx


def eval_method(ev: InstanceEvaluator, method: str, out_dir: str) -> None:
    out_path = os.path.join(out_dir, f"{method}.json")
    if os.path.exists(out_path):
        print(f"skip {method}: exists")
        return
    score_path = os.path.join(SCORES_DIR, ev.inst["instance_id"], f"{method}.json")
    if not os.path.exists(score_path):
        print(f"skip {method}: no scores at {score_path}")
        return
    with open(score_path) as f:
        scores = json.load(f)["scores"]
    order = ranking_from_scores(scores, ev.num_compress)

    rows = []
    for M in m_list_for(ev.num_compress):
        selected = sorted(order[:M])
        rows.append({
            "M": M,
            "keep_fraction": M / ev.num_compress,
            "their_m": M in M_VALUES,
            "selected": selected,
            "answer_kl": ev.answer_kl(selected),
            "token_kl": ev.token_kl(selected),
        })
        print(f"  {method} M={M}: answer_kl={rows[-1]['answer_kl']:.4f} "
              f"token_kl={rows[-1]['token_kl']}")
    _dump(out_path, ev, method, rows)


def eval_oracle(ev: InstanceEvaluator, out_dir: str) -> None:
    out_path = os.path.join(out_dir, "oracle.json")
    if os.path.exists(out_path):
        print("skip oracle: exists")
        return
    rows = []
    for M in m_list_for(ev.num_compress):
        backfill = list(range(ev.num_compress - (M - 1), ev.num_compress))
        candidates = [i for i in range(ev.num_compress) if i not in backfill]
        best = None
        cand_kls = {}
        for c in candidates:
            selected = sorted(set([c] + backfill))
            kl = ev.answer_kl(selected)
            cand_kls[c] = kl
            if best is None or kl < best[1]:
                best = (c, kl)
        selected = sorted(set([best[0]] + backfill))
        rows.append({
            "M": M,
            "keep_fraction": M / ev.num_compress,
            "their_m": M in M_VALUES,
            "selected": selected,
            "best_candidate": best[0],
            "answer_kl": best[1],
            "token_kl": ev.token_kl(selected),
            "candidate_kls": cand_kls,
        })
        print(f"  oracle M={M}: best={best[0]} answer_kl={best[1]:.4f}")
    _dump(out_path, ev, "oracle", rows)


def eval_reference(ev: InstanceEvaluator, out_dir: str) -> None:
    """Deletion (M=0) and recomputed-baseline sanity rows."""
    out_path = os.path.join(out_dir, "reference.json")
    if os.path.exists(out_path):
        print("skip reference: exists")
        return
    deletion_kl = ev.answer_kl([])
    deletion_token_kl = ev.token_kl([])
    recomputed = ev.answer_dist(list(range(ev.num_compress)))
    drift = kl_from_distributions(ev.baseline_dist, recomputed)
    rows = [
        {"M": 0, "row": "deletion", "answer_kl": deletion_kl,
         "token_kl": deletion_token_kl},
        {"row": "baseline_recompute_drift", "answer_kl": drift},
    ]
    print(f"  reference: deletion_kl={deletion_kl:.4f} baseline_drift={drift:.2e}")
    _dump(out_path, ev, "reference", rows)


def _dump(out_path: str, ev: InstanceEvaluator, method: str, rows: list) -> None:
    with open(out_path, "w") as f:
        json.dump({
            "instance_id": ev.inst["instance_id"],
            "question_id": ev.qid,
            "prefix_length": ev.N,
            "num_rankable": ev.num_compress,
            "bucket": ev.inst["bucket"],
            "method": method,
            "letters": ev.letters,
            "baseline_dist": ev.baseline_dist,
            "rows": rows,
        }, f, indent=2)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument(
        "--methods",
        default="reference,suppress,ta,attn_last,attn_next,oracle",
    )
    parser.add_argument("--model_name", default=MODEL_NAME)
    args = parser.parse_args()

    with open(os.path.join(DATA_DIR, "instances.json")) as f:
        instances = {r["instance_id"]: r for r in json.load(f)}
    inst = instances[args.instance]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    run(inst, model, tokenizer, args.methods.split(","))


def run(inst: dict, model, tokenizer, methods: List[str]) -> None:
    ev = InstanceEvaluator(inst, model, tokenizer)
    out_dir = os.path.join(EVALS_DIR, inst["instance_id"])
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for method in methods:
        if method == "oracle":
            eval_oracle(ev, out_dir)
        elif method == "reference":
            eval_reference(ev, out_dir)
        else:
            eval_method(ev, method, out_dir)
    print(f"instance {inst['instance_id']} evals done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
