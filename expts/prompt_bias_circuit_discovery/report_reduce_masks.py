"""Summary table of ``eval_reduce_masks.py`` outputs: per pool x cut x
budget, the mean P(trace's answer) with no mask, under the learned mask,
under random removals of the same size and with the whole pool removed;
how many traces the learned mask moves by more than 0.1 beyond the random
baseline; and, for the prompt-to-reasoning pool, which prompt chunks the
learned mask removes reads of.

Usage: uv run python -m expts.prompt_bias_circuit_discovery.report_reduce_masks --name reduce_masks_qwen3_8b --md /tmp/x.md
"""

from __future__ import annotations

import argparse
import collections
import glob
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="reduce_masks_qwen3_8b")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    res = f"results/prompt_bias_v2/reduce_masks/{args.name}"
    E = [json.load(open(f)) for f in sorted(glob.glob(f"{res}/eval/*.json"))]
    D = {d["example_id"]: d for d in json.load(open(f"{res}/dataset.json"))}
    md = [f"{len(E)} evaluated masks of {len(D) * 8} ({len({e['example_id'] for e in E})} traces).\n",
          "| pool | cut | removed | group | n | P(trace), no mask | learned | random (mean of 5) | whole pool removed | learned minus random (mean) | traces with learned below random by > 0.1 | learned below whole-pool by > 0.05 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = [("all", lambda e: True), ("effect inputs, admit trace", lambda e: e["is_positive"] and e["trace_answer"] == "A"),
              ("effect inputs, reject trace", lambda e: e["is_positive"] and e["trace_answer"] == "B"),
              ("null inputs, admit trace", lambda e: (not e["is_positive"]) and e["trace_answer"] == "A"),
              ("null inputs, reject trace", lambda e: (not e["is_positive"]) and e["trace_answer"] == "B")]
    for pool in ["prompt_to_trace", "trace_to_trace"]:
        for cut in ["half", "think"]:
            for sp in sorted({e["target_sparsity"] for e in E}):
                for gname, gf in groups:
                    sel = [e for e in E if e["pool"] == pool and e["cut"] == cut and e["target_sparsity"] == sp and gf(e)]
                    if not sel:
                        continue
                    c = np.array([e["clean"]["p_target"] for e in sel]); l = np.array([e["learned"]["p_target"] for e in sel])
                    r = np.array([np.mean([x["p_target"] for x in e["random"]]) for e in sel]); a = np.array([e["all_removed"]["p_target"] for e in sel])
                    md.append(f"| {pool} | {cut} | {int(sp * 100)}% | {gname} | {len(sel)} | {c.mean():.3f} | {l.mean():.3f} | {r.mean():.3f} | {a.mean():.3f} | {(l - r).mean():+.3f} | "
                              f"{int(((r - l) > 0.1).sum())} | {int(((a - l) > 0.05).sum())} |")
    # what the prompt-to-reasoning masks remove
    md.append("\nReads removed by the learned prompt-to-reasoning masks, by key chunk kind (summed over traces), and reads of the name chunks removed:\n")
    md.append("| cut | removed | key chunk kind: cells removed | name-chunk cells removed / traces | pool size (mean) |")
    md.append("|---|---|---|---|---|")
    for cut in ["half", "think"]:
        for sp in sorted({e["target_sparsity"] for e in E}):
            sel = [e for e in E if e["pool"] == "prompt_to_trace" and e["cut"] == cut and e["target_sparsity"] == sp]
            if not sel:
                continue
            kinds = collections.Counter()
            for e in sel:
                kinds.update(e["removed_cells_by_key_kind"])
            md.append(f"| {cut} | {int(sp * 100)}% | {dict(kinds)} | {sum(e['removed_name_cells'] for e in sel)} / {len(sel)} | {np.mean([e['n_pool'] for e in sel]):.0f} |")
    # per-trace table
    md.append("\n<details><summary>Per trace</summary>\n")
    md.append("| trace | input | answer | pool | cut | removed | P(trace) no mask | learned | random | whole pool |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for e in sorted(E, key=lambda e: (e["example_id"], e["pool"], e["cut"], e["target_sparsity"])):
        md.append(f"| {e['tag']} | {'effect' if e['is_positive'] else 'null'} | {e['trace_answer']} | {e['pool']} | {e['cut']} | {int(e['target_sparsity'] * 100)}% | {e['clean']['p_target']:.3f} | "
                  f"{e['learned']['p_target']:.3f} | {np.mean([x['p_target'] for x in e['random']]):.3f} | {e['all_removed']['p_target']:.3f} |")
    md.append("\n</details>")
    text = "\n".join(md); print(text[:3000])
    if args.md:
        open(args.md, "w").write(text)


if __name__ == "__main__":
    main()
