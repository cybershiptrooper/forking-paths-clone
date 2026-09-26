"""Run every chunking method over every prompt family and record whether
the cue shares a chunk with other task information.

Output: results/prompt_bias_v2/analysis/confound_sweep.json with, per
family and method, the fraction of prompts whose cue chunks are all
clean, the mean number of prompt chunks, and per-prompt details (used by
the notebook to display the chunking of random prompts).

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.confound_sweep --out results/prompt_bias_v2/analysis/confound_sweep.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random

import numpy as np
from transformers import AutoTokenizer

from expts.prompt_bias_circuit_discovery.prompt_chunking import chunk_prompt, cue_char_spans, confound_report
from expts.prompt_bias_circuit_discovery.openrouter_client import get_client, load_cache

FAMILIES = {
    "resume": ("results/prompt_bias/prompts.json", lambda p: p["axis"] != "neutral"),
    "discrim_mp_explicit": ("results/prompt_bias_v2/discrim_mp_prompts.json", lambda p: p["fill_type"] == "explicit"),
    "discrim_mp_implicit": ("results/prompt_bias_v2/discrim_mp_prompts.json", lambda p: p["fill_type"] == "implicit"),
    "bbq": ("results/prompt_bias/bbq_prompts.json", lambda p: True),
}
ORDER = ["sentence", "llm", "content", "token"]  # coarse to fine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_family", type=int, default=120)
    ap.add_argument("--llm_model", default="google/gemini-3.8-flash")
    ap.add_argument("--seg_cache", default="results/prompt_bias_v2/llm_seg_cache.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_workers", type=int, default=16)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    llm = (get_client(), args.llm_model, load_cache(args.seg_cache), args.seg_cache)
    rng = random.Random(args.seed)
    out = {"families": {}, "details": {}}
    from concurrent.futures import ThreadPoolExecutor
    from expts.prompt_bias_circuit_discovery.prompt_chunking import seg_llm
    for fam, (path, keep) in FAMILIES.items():
        P = [p for p in json.load(open(path)) if keep(p)]
        rng.shuffle(P)
        P = P[:args.n_per_family]
        # warm the LLM segmentation cache concurrently (seg_llm is deterministic and cached by text)
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            list(ex.map(lambda p: seg_llm(p["question"], *llm), P))
        stats = collections.defaultdict(list)
        details = []
        for p in P:
            cue = cue_char_spans(p)
            d = dict(uid=p["uid"], cue=[p["question"][s:e] for s, e in cue], methods={})
            for m in ORDER:
                chunks, info = chunk_prompt(tok, p, m, llm=llm, forced_spans=cue)
                rep = confound_report(tok, p, chunks, cue)
                stats[m].append((rep["all_clean"], rep["n_chunks"], info.get("llm_status", "")))
                d["methods"][m] = dict(n_chunks=rep["n_chunks"], all_clean=rep["all_clean"], status=info.get("llm_status", ""),
                                       chunks=[(c["kind"], c["text"]) for c in chunks],
                                       cue_chunks=[(r["text"], r["residual_content"], r["residual_role"]) for r in rep["cue_chunks"]])
            details.append(d)
        out["details"][fam] = details
        out["families"][fam] = {m: dict(n=len(v), frac_clean=float(np.mean([a for a, _, _ in v])), mean_chunks=float(np.mean([n for _, n, _ in v])),
                                        n_llm_fallback=sum(1 for _, _, s in v if "fallback" in s))
                                for m, v in stats.items()}
        best = next((m for m in ORDER if out["families"][fam][m]["frac_clean"] == 1.0), "token")
        out["families"][fam]["best"] = best
        print(fam, {m: (round(s["frac_clean"], 3), round(s["mean_chunks"], 1)) for m, s in out["families"][fam].items() if m != "best"}, "best:", best, flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print("saved", args.out)


if __name__ == "__main__":
    main()
