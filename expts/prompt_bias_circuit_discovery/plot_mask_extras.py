"""Figures for the mask-assisted judge note that are not per-class results:
the family-selection baselines and the block-variant tuning run.

Usage: uv run python -m expts.prompt_bias_circuit_discovery.plot_mask_extras
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from expts.prompt_bias_circuit_discovery.judge_bias_v2 import auroc, auroc_ci
from expts.prompt_bias_circuit_discovery.report_masks import COND_LABEL, COLORS, JUDGE_LABEL

IMG = "notes/images/prompt_bias_masks"
FAM = {"discrim_mp_implicit": "discrim-eval names", "discrim_mp_explicit": "discrim-eval race word", "resume": "resume",
       "karvonen_hiring": "Karvonen hiring", "blindspot_admission": "blind-spot admission", "blindspot_loan": "blind-spot loan"}


def fig_selection():
    rep = json.load(open("results/prompt_bias_masks/selection_report_qwen3_8b.json"))
    ext = {}
    for c in ["prompt_change", "same_prompt"]:
        for j in ["gpt56sol", "gemini38flash"]:
            f = f"results/prompt_bias_masks/judge/pool_ext_discrim_mp_explicit_{c}_{j}.json"
            if os.path.exists(f):
                it = [i for i in json.load(open(f))["items"] if i["condition"] == "baseline" and i.get("probability") is not None]
                ext[f"{c}|{j}"] = auroc([i["probability"] for i in it if i["is_positive"]], [i["probability"] for i in it if not i["is_positive"]])
    fams = ["discrim_mp_implicit", "discrim_mp_explicit", "resume", "karvonen_hiring", "blindspot_admission", "blindspot_loan"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, c in zip(axes, ["prompt_change", "same_prompt"]):
        for k, j in enumerate(["gpt56sol", "gemini38flash"]):
            vals = [rep["family_auc"][f"{c}|{j}"].get(f, [None])[0] if f != "discrim_mp_explicit" else ext.get(f"{c}|{j}") for f in fams]
            ax.bar(np.arange(len(fams)) + (k - 0.5) * 0.38, [v or 0 for v in vals], 0.38, label=JUDGE_LABEL[j], color=["#4c72b0", "#dd8452"][k])
        ax.axhline(0.9, ls="--", color="r", lw=1, label="saturated (0.9)"); ax.axhline(0.5, ls=":", color="k", lw=0.8)
        ax.set_xticks(range(len(fams))); ax.set_xticklabels([FAM[f] for f in fams], rotation=25, ha="right", fontsize=9)
        ax.set_title(f"{c}: baseline AUROC per family (candidate pool)"); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("AUROC"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{IMG}/selection_family_auroc.png", dpi=130); plt.close(fig)


def fig_variants():
    V = [("k25", "25 pairs"), ("k8", "8 pairs"), ("k50", "50 pairs"), ("k0", "no pairs"), ("top5", "top 5 segments"), ("text", "+ text"),
         ("guid", "+ guidance"), ("guidtext", "guidance + text"), ("only", "analysis only")]
    conds = ["mask:attribution", "mask:p2t_rg", "mask:ta", "mask:p2t_kl"]
    judges = ["gpt56sol", "gemini38flash"]; classes = ["explicit", "prompt_change", "same_prompt"]
    fig, axes = plt.subplots(len(judges), len(classes), figsize=(16, 7), sharey=True)
    w = 0.8 / len(conds)
    for r, j in enumerate(judges):
        base = {i["tag"] for i in json.load(open(f"results/prompt_bias_masks/judge/dataset_with_blocks_qwen3_8b_k25_{j}.json"))["items"]}
        for c, cls in enumerate(classes):
            ax = axes[r, c]
            ref = {}
            for vi, (v, vl) in enumerate(V):
                f = f"results/prompt_bias_masks/judge/dataset_with_blocks_qwen3_8b_{v}_{j}.json"
                if not os.path.exists(f):
                    continue
                items = [i for i in json.load(open(f))["items"] if i["case"] == cls and i["tag"] in base and i.get("probability") is not None]
                for ci, cond in enumerate(["baseline", "ceiling"] + conds):
                    it = [i for i in items if i["condition"] == cond]
                    pos = [i["probability"] for i in it if i["is_positive"]]; neg = [i["probability"] for i in it if not i["is_positive"]]
                    if not pos or not neg:
                        continue
                    a = auroc(pos, neg)
                    if cond in ("baseline", "ceiling"):
                        ref.setdefault(cond, a); continue
                    lo, hi = auroc_ci(pos, neg, n_boot=500)
                    k = conds.index(cond)
                    ax.bar(vi + (k - len(conds) / 2 + 0.5) * w, a, w, yerr=[[a - lo], [hi - a]], color=COLORS[cond], capsize=1.5,
                           error_kw=dict(lw=0.6), label=COND_LABEL[cond] if (r, c, vi) == (0, 0, 0) else None)
            for cond, ls in [("baseline", ":"), ("ceiling", "--")]:
                if cond in ref:
                    ax.axhline(ref[cond], ls=ls, color="k", lw=1, label=COND_LABEL[cond] if (r, c) == (0, 0) else None)
            ax.set_xticks(range(len(V))); ax.set_xticklabels([vl for _, vl in V], rotation=35, ha="right", fontsize=8)
            ax.set_ylim(0, 1.05); ax.set_title(f"{JUDGE_LABEL[j]}: {cls}", fontsize=10)
    axes[0, 0].set_ylabel("AUROC"); axes[1, 0].set_ylabel("AUROC")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{IMG}/block_variants.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(IMG, exist_ok=True)
    fig_selection(); fig_variants()
    print("figures written")
