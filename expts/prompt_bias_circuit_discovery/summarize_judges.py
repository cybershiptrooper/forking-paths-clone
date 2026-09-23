"""Markdown table of judge AUROCs (with bootstrap intervals) over every
judge result file, one row per (traces model, family, case) and one
column per (judge, condition).

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.summarize_judges > /tmp/judge_table.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

JUDGE_LABEL = {"gpt56luna": "GPT-5.6 Luna", "gemini38flash": "Gemini 3.8 Flash", "qwen32b": "Qwen3-32B (P(yes))",
               "gpt56luna_noreason": "GPT-5.6 Luna", "gemini38flash_noreason": "Gemini 3.8 Flash", "qwen32b_noreason": "Qwen3-32B (P(yes))"}
FAM_LABEL = {"resume": "resume", "discrim_mp_explicit": "discrim explicit", "discrim_mp_implicit": "discrim implicit", "bbq": "BBQ",
             "karvonen_hiring": "Karvonen hiring", "blindspot_loan": "blind-spot loan", "blindspot_admission": "blind-spot admission"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_dir", default="results/prompt_bias_v2/judge")
    ap.add_argument("--json_out", default="results/prompt_bias_v2/analysis/judge_summary.json")
    args = ap.parse_args()
    rows = {}
    for f in sorted(glob.glob(os.path.join(args.judge_dir, "*.json"))):
        b = os.path.basename(f)
        m = re.match(r"(qwen3_\d+b)_(.+?)_(prompt_change_same_answer|prompt_change|same_prompt|explicit_control)_(.+)\.json", b)
        if not m:
            continue
        model, fam, case, judge = m.groups()
        d = json.load(open(f))
        key = "p_yes" if any(k.endswith("|p_yes") for k in d["summary"]) else "probability"
        from expts.prompt_bias_circuit_discovery.judge_bias_v2 import tpr_at_fpr
        for k, v in d["summary"].items():
            case2, cond, kk = k.split("|")
            if kk != key:
                continue
            its = [i for i in d["items"] if i["case"] == case2 and i["condition"] == cond and i.get(key) is not None]
            v = dict(v)
            v["tpr1"] = tpr_at_fpr([i[key] for i in its if i["is_positive"]], [i[key] for i in its if not i["is_positive"]], 0.01)
            rows.setdefault((model, fam, case), {})[(judge, cond)] = v
    judges = ["gpt56luna", "gemini38flash", "qwen32b"]
    conds = ["baseline", "ceiling", "no_reasoning"]
    # the no_reasoning runs are stored under <judge>_noreason tags; fold them in
    for (model, fam, case), d in rows.items():
        for (judge, cond), v in list(d.items()):
            if judge.endswith("_noreason"):
                d[(judge[:-len("_noreason")], "no_reasoning")] = v
    # the reasoning increment: AUROC with the reasoning (baseline) minus without it (no_reasoning);
    # what the judge gets from the chain of thought beyond the question and the answer
    out_lines = ["| traces | family | case | n pos/neg | " + " | ".join(f"{JUDGE_LABEL[j]} {c}" for j in judges for c in conds)
                 + " | " + " | ".join(f"{JUDGE_LABEL[j]} increment" for j in judges)
                 + " | " + " | ".join(f"{JUDGE_LABEL[j]} TPR@1%FPR (baseline / no reasoning)" for j in judges) + " |",
                 "|---|---|---|---|" + "---|" * (len(judges) * len(conds) + 2 * len(judges))]
    for (model, fam, case) in sorted(rows, key=lambda t: (t[0], list(FAM_LABEL).index(t[1]) if t[1] in FAM_LABEL else 9, ["prompt_change", "prompt_change_same_answer", "same_prompt", "explicit_control"].index(t[2]))):
        cells = []
        n = None
        for j in judges:
            for c in conds:
                v = rows[(model, fam, case)].get((j, c))
                if v is None or v["auroc"] is None:
                    cells.append("-")
                else:
                    n = n or f"{v['n_pos']}/{v['n_neg']}"
                    cells.append(f"{v['auroc']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]")
        incs = []
        for j in judges:
            b = rows[(model, fam, case)].get((j, "baseline")); nr = rows[(model, fam, case)].get((j, "no_reasoning"))
            incs.append(f"{b['auroc'] - nr['auroc']:+.2f}" if b and nr and b["auroc"] is not None and nr["auroc"] is not None else "-")
        tprs = []
        for j in judges:
            b = rows[(model, fam, case)].get((j, "baseline")); nr = rows[(model, fam, case)].get((j, "no_reasoning"))
            fb = f"{b['tpr1'][0]:.2f}" if b and b["tpr1"][0] is not None else "-"
            fn = f"{nr['tpr1'][0]:.2f}" if nr and nr["tpr1"][0] is not None else "-"
            tprs.append(f"{fb} / {fn}")
        out_lines.append(f"| {model} | {FAM_LABEL.get(fam, fam)} | {case} | {n} | " + " | ".join(cells) + " | " + " | ".join(incs) + " | " + " | ".join(tprs) + " |")
    print("\n".join(out_lines))
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    json.dump({f"{m}|{f}|{c}": {f"{j}|{cond}": v for (j, cond), v in d.items()} for (m, f, c), d in rows.items()}, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
