"""Tables and figures for ``eval_prompt_chunk_text_dependence.py`` on the
same-decision admission set: does the reasoning text of an admit trace depend
on the applicant's name chunk more when the name moved the admit rate
(positives) than when it did not (negatives), with the same trace under the
White-name prompt as the within-trace control?

Usage: uv run python -m expts.prompt_bias_circuit_discovery.report_text_dependence --group black --md /tmp/x.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

from expts.prompt_bias_circuit_discovery.judge_bias_v2 import auroc


def name_stats(arm):
    """Per-arm statistics of the name chunk's effect on the reasoning text."""
    j = "name_all"
    has = arm["sentence_has_name"]
    ntok = np.array(arm["sentence_n_tokens"], dtype=float)
    kl_name = np.array([x if x is not None else np.nan for x in arm["chunks"][j]["sentence_kl"]])
    keep = np.array([not h and n > 0 for h, n in zip(has, ntok)])
    w = ntok * keep
    tok_mean = lambda v: float(np.nansum(v * w) / max(1e-9, w.sum()))
    name_set = set(arm["name_chunks"])
    q_idx = [k for k, kind in enumerate(arm["chunk_kinds"]) if kind == "question"]
    per_chunk = {}
    for k in q_idx:
        v = np.array([x if x is not None else np.nan for x in arm["chunks"][str(k)]["sentence_kl"]])
        per_chunk[k] = tok_mean(v)
    name_val = tok_mean(kl_name)  # every mention of the name removed at once
    others = [v for k, v in per_chunk.items() if k not in name_set]
    rank = 1 + sum(v > name_val for v in others)  # 1 = the name matters more for the text than any other application chunk
    share = name_val / max(1e-9, name_val + sum(others))
    kl_all = np.array([x if x is not None else np.nan for x in arm["chunks"]["all"]["sentence_kl"]])
    kl_unread = np.array([x if x is not None else np.nan for x in arm["chunks"]["name_all_unread"]["sentence_kl"]])
    cuts = {c: {k: v for k, v in row.items()} for c, row in arm["cuts"].items()}
    return dict(name_kl=name_val, name_kl_incl=float(np.nansum(kl_name * ntok) / ntok.sum()), name_rank=rank, name_share=share, n_question_chunks=len(q_idx),
                n_name_mentions=len(name_set), all_kl=tok_mean(kl_all), unread_kl=tok_mean(kl_unread), max_other=max(others) if others else np.nan,
                mean_other=float(np.mean(others)) if others else np.nan, per_chunk=per_chunk,
                profile=[(float(kl_name[k]) if keep[k] else None) for k in range(len(has))],
                p_admit_clean=arm["p_admit_clean"], p_admit_no_name=arm["chunks"][j]["p_admit"], p_admit_unread=arm["chunks"]["name_all_unread"]["p_admit"],
                p_admit_no_prompt=arm["chunks"]["all"]["p_admit"], cuts=cuts,
                n_name_sentences=int(sum(has)), n_sentences=len(has))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="black")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--img_dir", default="notes/images/prompt_bias_dataset")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    root = "results/prompt_bias_v2/same_decision_sweep"
    D = {r["example_id"]: r for r in json.load(open(f"{root}/dataset_{args.tag}_{args.group}.json"))}
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/text_dependence_v2_{args.tag}_{args.group}/ex*.json"))]
    ex = []
    for r in R:
        b, w = name_stats(r["arms"]["black"]), name_stats(r["arms"]["white"])
        m = D[r["example_id"]]
        ex.append(dict(id=r["example_id"], pos=r["is_positive"], b=b, w=w, diff=b["name_kl"] - w["name_kl"], ratio=b["name_kl"] / max(1e-9, w["name_kl"]),
                       attributable=m.get("attributable_white"), delta=m.get("delta_white"), retok=r.get("retokenised_equal")))
    P = [e for e in ex if e["pos"]]; N = [e for e in ex if not e["pos"]]
    md = [f"{len(ex)} examples ({len(P)} positives, {len(N)} negatives); retokenised reasoning identical to the stored tokens in {sum(bool(e['retok']) for e in ex)} of {len(ex)}. "
          f"Sentences naming the applicant (excluded from every text statistic): median {np.median([e['b']['n_name_sentences'] for e in ex]):.0f} of {np.median([e['b']['n_sentences'] for e in ex]):.0f} per trace. "
          f"Question chunks per prompt: {ex[0]['b']['n_question_chunks']} (the name is one of them)."]
    md.append("\n| statistic (Black-name arm unless stated) | positives: median [IQR] | negatives: median [IQR] | AUROC pos vs neg | Mann-Whitney p |")
    md.append("|---|---|---|---|---|")
    stats = [("name (every mention): mean KL per reasoning token when the reasoning's reads of it are removed", lambda e: e["b"]["name_kl"]),
             ("name: the same with every prompt token's reads of it removed too (name unread)", lambda e: e["b"]["unread_kl"]),
             ("name: rank among the application chunks (1 = largest KL)", lambda e: e["b"]["name_rank"]),
             ("name: share of the summed per-chunk KL", lambda e: e["b"]["name_share"]),
             ("mean over the other application chunks: KL per token when one chunk's reads are removed", lambda e: e["b"]["mean_other"]),
             ("all prompt chunks removed: mean KL per token", lambda e: e["b"]["all_kl"]),
             ("name KL, White-name arm (same trace, name swapped)", lambda e: e["w"]["name_kl"]),
             ("name KL, Black minus White arm", lambda e: e["diff"]),
             ("P(admit) drop at </think>, reasoning's reads of the name removed", lambda e: e["b"]["p_admit_clean"] - e["b"]["p_admit_no_name"]),
             ("P(admit) drop at </think>, name unread", lambda e: e["b"]["p_admit_clean"] - e["b"]["p_admit_unread"]),
             ("P(admit) drop at </think>, every prompt read removed", lambda e: e["b"]["p_admit_clean"] - e["b"]["p_admit_no_prompt"])]
    summary = {}
    for lab, f in stats:
        p = [f(e) for e in P]; n = [f(e) for e in N]
        a = auroc(p, n); pv = mannwhitneyu(p, n).pvalue
        q = lambda v: f"{np.median(v):.4f} [{np.percentile(v, 25):.4f}, {np.percentile(v, 75):.4f}]"
        md.append(f"| {lab} | {q(p)} | {q(n)} | {a:.2f} | {pv:.2g} |")
        summary[lab] = dict(auroc=a, p=pv)
    # within-trace control: is Black > White within class?
    for lab, sel in [("positives", P), ("negatives", N)]:
        d = [e["diff"] for e in sel]
        md.append(f"\nBlack minus White name-chunk KL, {lab}: mean {np.mean(d):+.4f}, positive in {sum(x > 0 for x in d)} of {len(d)}, Wilcoxon p = {wilcoxon(d).pvalue:.2g}.")
    rho = spearmanr([e["attributable"] for e in P], [e["b"]["name_kl"] for e in P])
    rho2 = spearmanr([e["attributable"] for e in P], [e["diff"] for e in P])
    md.append(f"\nAmong the positives, Spearman correlation of the attributable share with the name chunk KL: {rho.correlation:+.2f} (p = {rho.pvalue:.2g}); with the Black minus White difference: {rho2.correlation:+.2f} (p = {rho2.pvalue:.2g}).")
    # P(admit) by cut: both arms, name unread
    md.append("\n### P(admit) at the probe by cut: the name swap and the name-unread ablation on the same trace prefix\n")
    md.append("| cut | class | n | Black name, no mask | White name, no mask (same trace) | Black minus White (examples with a drop > 0.1) | AUROC of the swap effect | Black, reasoning's reads of the name removed (change) | Black, name unread (change; drop > 0.1) | AUROC of the unread drop |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    CUTN = {"prompt": "prompt only, no reasoning", "0.25": "0.25 of the reasoning", "0.5": "0.5 of the reasoning", "0.75": "0.75 of the reasoning", "before_decision": "just before the decision sentence", "think": "at </think>"}
    cut_summary = {}
    for c in ["prompt", "0.25", "0.5", "0.75", "before_decision", "think"]:
        swap_all = {}; unread_all = {}
        for cls, sel in [("positives", P), ("negatives", N)]:
            pb = np.array([e["b"]["cuts"][c]["clean"] for e in sel]); pw = np.array([e["w"]["cuts"][c]["clean"] for e in sel])
            un = np.array([e["b"]["cuts"][c]["name_all_unread"] for e in sel])
            na = np.array([e["b"]["cuts"][c].get("name_all", np.nan) for e in sel])
            swap_all[cls] = pb - pw; unread_all[cls] = pb - un
            cut_summary[(c, cls)] = dict(black=float(pb.mean()), white=float(pw.mean()), unread=float(un.mean()), swap=float((pb - pw).mean()))
            na_txt = "" if np.isnan(na).all() else f"{(na - pb).mean():+.3f}"
            md.append(f"| {CUTN[c]} | {cls} | {len(sel)} | {pb.mean():.3f} | {pw.mean():.3f} | {(pb - pw).mean():+.3f} ({int(((pb - pw) > 0.1).sum())}) | | {na_txt} | {(un - pb).mean():+.3f} ({int(((pb - un) > 0.1).sum())}) | |")
        a_swap = auroc(list(swap_all["positives"]), list(swap_all["negatives"])); a_un = auroc(list(unread_all["positives"]), list(unread_all["negatives"]))
        md[-2] = md[-2].replace("| | ", f"| {a_swap:.2f} | ", 1).replace(" | |", f" | {a_un:.2f} |", 1)
        md[-1] = md[-1].replace("| | ", f"| {a_swap:.2f} | ", 1).replace(" | |", f" | {a_un:.2f} |", 1)
    rho3 = spearmanr([e["attributable"] for e in P], [e["b"]["cuts"]["prompt"]["clean"] - e["w"]["cuts"]["prompt"]["clean"] for e in P])
    md.append(f"\nAmong the positives, Spearman correlation of the attributable share with the swap effect on the prompt-only forced answer: {rho3.correlation:+.2f} (p = {rho3.pvalue:.2g}).")

    # ---- figures ----
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    ax = axes[0]
    for sel, col, lab in [(N, "#2166ac", "negatives"), (P, "#d6604d", "positives")]:
        ax.scatter([e["w"]["name_kl"] for e in sel], [e["b"]["name_kl"] for e in sel], s=14, color=col, alpha=0.7, label=f"{lab} ({len(sel)})")
    lim = max(max(e["b"]["name_kl"], e["w"]["name_kl"]) for e in ex) * 1.05
    ax.plot([0, lim], [0, lim], ":", color="k", lw=0.8); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("name chunk KL, same trace under the White name"); ax.set_ylabel("name chunk KL, Black name")
    ax.set_title("the name chunk's effect on the reasoning text, both arms", fontsize=10); ax.legend(fontsize=8)
    ax = axes[1]
    rng = np.random.default_rng(0)
    for i_, (lab, f) in enumerate([("name KL (Black)", lambda e: e["b"]["name_kl"]), ("Black minus White", lambda e: e["diff"]), ("name rank", lambda e: e["b"]["name_rank"])]):
        pass
    for i_, (lab, f) in enumerate([("name KL, Black", lambda e: e["b"]["name_kl"]), ("Black minus White", lambda e: e["diff"])]):
        for sel, col, dx in [(P, "#d6604d", -0.17), (N, "#2166ac", 0.17)]:
            v = [f(e) for e in sel]
            ax.boxplot([v], positions=[i_ + dx], widths=0.28, showfliers=False, patch_artist=True, boxprops=dict(facecolor=col, alpha=0.4), medianprops=dict(color="k"))
            ax.scatter(i_ + dx + rng.uniform(-0.07, 0.07, len(v)), v, s=6, color=col, alpha=0.6, zorder=3)
    ax.axhline(0, color="k", ls=":", lw=0.6); ax.set_xticks([0, 1]); ax.set_xticklabels(["name KL, Black name", "Black minus White"], fontsize=9)
    ax.set_ylabel("mean KL per reasoning token"); ax.set_title("per example (left box: positives, right: negatives)", fontsize=10)
    ax = axes[2]
    bins = np.linspace(0, 1, 11)
    for sel, col, lab in [(N, "#2166ac", "negatives"), (P, "#d6604d", "positives")]:
        prof = [[] for _ in range(10)]
        for e in sel:
            pr = e["b"]["profile"]; n = len(pr)
            for k, v in enumerate(pr):
                if v is not None:
                    prof[min(9, int(k / n * 10))].append(v)
        ax.plot((bins[:-1] + bins[1:]) / 2, [np.mean(p) if p else np.nan for p in prof], "-o", color=col, ms=4, label=lab)
    ax.set_xlabel("position of the reasoning sentence (fraction of the trace)"); ax.set_ylabel("mean KL per token, name chunk removed")
    ax.set_title("where in the trace the name chunk matters (Black name)", fontsize=10); ax.legend(fontsize=8)
    ax = axes[3]
    order = ["prompt", "0.25", "0.5", "0.75", "before_decision", "think"]
    for cls, col in [("positives", "#d6604d"), ("negatives", "#2166ac")]:
        ax.plot(range(len(order)), [cut_summary[(c, cls)]["black"] for c in order], "-o", color=col, ms=4, lw=2, label=f"{cls}, Black name")
        ax.plot(range(len(order)), [cut_summary[(c, cls)]["white"] for c in order], "--s", color=col, ms=4, lw=1.4, label=f"{cls}, White name (same trace)")
        ax.plot(range(len(order)), [cut_summary[(c, cls)]["unread"] for c in order], ":^", color=col, ms=4, lw=1.4, label=f"{cls}, Black name unread")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(["prompt only", "0.25", "0.5", "0.75", "before decision", "</think>"], fontsize=8, rotation=20)
    ax.set_ylim(0.7, 1.0); ax.set_ylabel("mean P(admit) at the probe"); ax.legend(fontsize=6.5, loc="lower right"); ax.set_title("the name swap and the name-unread ablation, by cut", fontsize=10)
    fig.tight_layout(); os.makedirs(args.img_dir, exist_ok=True)
    f1 = f"{args.img_dir}/same_decision_textdep_{args.group}.png"; fig.savefig(f1, dpi=130); plt.close(fig); print("figure", f1)
    text = "\n".join(md); print(text)
    if args.md:
        open(args.md, "w").write(text)
    json.dump([{k: v for k, v in e.items() if k not in ("b", "w")} | {"black": {k: v for k, v in e["b"].items() if k not in ("profile", "per_chunk")}, "white": {k: v for k, v in e["w"].items() if k not in ("profile", "per_chunk")}} for e in ex],
              open(f"{root}/text_dependence_v2_{args.tag}_{args.group}_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
