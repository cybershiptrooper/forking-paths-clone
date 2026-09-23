"""Tables and figures for the mask-assisted judge comparison.

Reads results/prompt_bias_masks/judge/dataset_with_blocks_<tag>_<judge>.json
(one file per judge; conditions baseline, ceiling, mask:<method>) and
writes, per judge and class, the number of examples judged, the number
the judge gets right at a probability threshold of 50, the AUROC with a
bootstrap interval and the TPR at 1 percent FPR (with fewer than 100
negatives: the TPR at zero false positives), as a markdown table
(``--md``), a JSON summary and figures under notes/images/prompt_bias_masks/.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.report_masks --tag qwen3_8b \
        --judges gpt56sol gemini38flash --md /tmp/mask_table.md
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from expts.prompt_bias_circuit_discovery.judge_bias_v2 import auroc, auroc_ci, tpr_at_fpr

JUDGE_LABEL = {"gpt56sol": "GPT-5.6 Sol", "gemini38flash": "Gemini 3.8 Flash", "qwen32b": "Qwen3-32B"}
COND_LABEL = {"baseline": "baseline (no analysis)", "ceiling": "bias-informed judge (told which characteristic)", "mask:attribution": "prompt attribution",
              "mask:p2t_rg": "prompt->trace SNP, raise P(answer)", "mask:ta": "prompt->trace Thought Anchors",
              "mask:p2t_kl": "prompt->trace SNP, keep KL", "mask:trace_rg": "reasoning-only SNP, raise P(answer) (to do)",
              "maskonly:attribution": "prompt attribution, reasoning withheld", "maskonly:p2t_rg": "prompt->trace SNP, raise P(answer), reasoning withheld",
              "maskonly:ta": "prompt->trace Thought Anchors, reasoning withheld", "maskonly:p2t_kl": "prompt->trace SNP, keep KL, reasoning withheld"}
CONDS = list(COND_LABEL)
CLASSES = ["explicit", "prompt_change", "same_prompt"]
COLORS = {"baseline": "#7f7f7f", "ceiling": "#bdbdbd", "mask:attribution": "#e08214", "mask:p2t_rg": "#2166ac",
          "mask:ta": "#d6604d", "mask:p2t_kl": "#1b7837", "mask:trace_rg": "#762a83",
          "maskonly:attribution": "#f4b97a", "maskonly:p2t_rg": "#7fa8d4", "maskonly:ta": "#ecab9f", "maskonly:p2t_kl": "#8fc79a"}


VARIANT = ""


def load(tag, judge, judge_dir):
    f = os.path.join(judge_dir, f"dataset_with_blocks_{tag}{VARIANT}_{judge}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    subset = SUBSET.get(tag, {})
    key = "p_yes" if judge == "qwen32b" else "probability"
    rows = collections.defaultdict(dict)  # (case, cond) -> {tag: (is_pos, score)}
    for it in d["items"]:
        if it.get(key) is None:
            continue
        rows[(it["case"], it["condition"])][it["tag"]] = (it["is_positive"], float(it[key]))
    return rows, key


SUBSET = {}


def load_subsets(tag):
    """tag -> subset name ('' for the selected set) from the dataset file."""
    f = f"results/prompt_bias_masks/dataset_{tag}.json"
    if os.path.exists(f):
        SUBSET[tag] = {r["tag"]: r.get("subset", "") or "" for r in json.load(open(f))}


args_fpr = 0.05


def stats(rows, case, cond, key, restrict=None):
    r = rows.get((case, cond), {})
    if restrict is not None:
        r = {t: v for t, v in r.items() if t in restrict}
    pos = [s for p, s in r.values() if p]; neg = [s for p, s in r.values() if not p]
    if not pos or not neg:
        return None
    thr = 0.5 if key == "p_yes" else 50.0
    lo, hi = auroc_ci(pos, neg)
    t, tlo, thi = tpr_at_fpr(pos, neg, args_fpr)
    return dict(n_pos=len(pos), n_neg=len(neg), correct_pos=sum(s >= thr for s in pos), correct_neg=sum(s < thr for s in neg),
                auroc=auroc(pos, neg), ci=[lo, hi], tpr1=t, tpr1_ci=[tlo, thi], mean_pos=float(np.mean(pos)), mean_neg=float(np.mean(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--judges", nargs="+", default=["gpt56sol", "gemini38flash", "qwen32b"])
    ap.add_argument("--judge_dir", default="results/prompt_bias_masks/judge")
    ap.add_argument("--img_dir", default="notes/images/prompt_bias_masks")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json_out", default="results/prompt_bias_masks/analysis/mask_judge_summary.json")
    ap.add_argument("--matched", type=int, default=1, help="restrict every condition to the examples that have all mask blocks judged")
    ap.add_argument("--min_n", type=int, default=5, help="AUROC/TPR of the matched comparison are shown only with at least this many positives and negatives")
    ap.add_argument("--examples_md", default=None, help="write a per-example table (score per condition) for --examples_case here")
    ap.add_argument("--examples_case", default="explicit")
    ap.add_argument("--variant", default="", help="block variant suffix of the judge files (e.g. _k8)")
    ap.add_argument("--marker", default="RESULTS", help="marker pair in the notes (<!-- X_START --> ... <!-- X_END -->)")
    ap.add_argument("--notes", default=None, help="insert the tables and figures between the RESULTS markers of this notes file")
    ap.add_argument("--status", default="", help="status line written above the tables in the notes")
    args = ap.parse_args()
    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    load_subsets(args.tag)
    global VARIANT
    VARIANT = args.variant
    summary = {}
    md = []
    for judge in args.judges:
        L = load(args.tag, judge, args.judge_dir)
        if L is None:
            continue
        rows, key = L
        md.append(f"\n**{JUDGE_LABEL.get(judge, judge)}** (score: {'P(yes) from the logits' if key == 'p_yes' else '0-100 probability'})\n")
        md.append("| class | condition | judged (pos/neg) | correct pos | correct neg | AUROC [95% CI] | TPR at 5% FPR [95% CI] |")
        md.append("|---|---|---|---|---|---|---|")
        for case in CLASSES:
            conds_present = [c for c in CONDS if (case, c) in rows]
            mask_conds = [c for c in conds_present if c.startswith("mask:")]
            # 1. the full class under the conditions that cover every example (baseline, attribute named):
            #    the selected set as designed, and the set with the extension families added
            sub = SUBSET.get(args.tag, {})
            exts = sorted({v for v in sub.values() if v})
            for cond in ("baseline", "ceiling"):
                for label, restrict_sub in [("selected set", {t for t, v in sub.items() if not v} if sub else None)] + \
                                           ([("with " + ", ".join(exts) + " added", None)] if exts else []):
                    s = stats(rows, case, cond, key, restrict_sub)
                    if s is None:
                        continue
                    summary[f"{judge}|{case}|{cond}|all" + ("" if label == "selected set" else "+ext")] = s
                    md.append(f"| {case} | {COND_LABEL.get(cond, cond)}, {label} | {s['n_pos']}/{s['n_neg']} | {s['correct_pos']}/{s['n_pos']} | {s['correct_neg']}/{s['n_neg']} | "
                              f"{s['auroc']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] | {s['tpr1']:.2f} [{s['tpr1_ci'][0]:.2f}, {s['tpr1_ci'][1]:.2f}] |")
            # 2. the matched comparison: the examples judged under every mask condition; AUROC and TPR only
            #    once the matched set has at least --min_n positives and negatives
            if not mask_conds:
                continue
            restrict = set.intersection(*[set(rows[(case, c)]) for c in mask_conds]) if args.matched else None
            groups = [("matched", restrict)]
            if exts and restrict is not None:
                groups.append(("matched, selected set only", {t for t in restrict if not sub.get(t)}))
            for glabel, gset in groups:
                for cond in conds_present:
                    s = stats(rows, case, cond, key, gset)
                    if s is None:
                        continue
                    summary[f"{judge}|{case}|{cond}" + ("" if glabel == "matched" else "|selected")] = s
                    enough = s["n_pos"] >= args.min_n and s["n_neg"] >= args.min_n
                    auc = f"{s['auroc']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]" if enough else "(too few)"
                    tpr = f"{s['tpr1']:.2f} [{s['tpr1_ci'][0]:.2f}, {s['tpr1_ci'][1]:.2f}]" if enough else "(too few)"
                    md.append(f"| {case} | {COND_LABEL.get(cond, cond)}, {glabel} | {s['n_pos']}/{s['n_neg']} | {s['correct_pos']}/{s['n_pos']} | {s['correct_neg']}/{s['n_neg']} | {auc} | {tpr} |")
        # figure: AUROC and TPR per class x condition. A class is drawn on its matched set (every condition on the
        # same examples) once that set has --min_n positives and negatives; before that only the full-set baseline
        # and attribute-named conditions are drawn (hatched).
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        w = 0.8 / len(CONDS)
        for ax, metric, lab in [(axes[0], "auroc", "AUROC"), (axes[1], "tpr1", "TPR at 5% FPR")]:
            for ci_, cond in enumerate(CONDS):
                for k, case in enumerate(CLASSES):
                    sm = summary.get(f"{judge}|{case}|{cond}|selected") or summary.get(f"{judge}|{case}|{cond}")
                    matched_ok = sm is not None and sm["n_pos"] >= args.min_n and sm["n_neg"] >= args.min_n
                    base_ok = any(summary.get(f"{judge}|{case}|{c}") is not None and summary[f"{judge}|{case}|{c}"]["n_pos"] >= args.min_n
                                  and summary[f"{judge}|{case}|{c}"]["n_neg"] >= args.min_n for c in CONDS if c.startswith("mask:"))
                    if matched_ok and base_ok:
                        s, hatch = sm, None
                    elif cond in ("baseline", "ceiling") and not base_ok and summary.get(f"{judge}|{case}|{cond}|all") is not None:
                        s, hatch = summary[f"{judge}|{case}|{cond}|all"], "//"
                    else:
                        continue
                    x = k + (ci_ - len(CONDS) / 2 + 0.5) * w
                    c = s["ci"] if metric == "auroc" else s["tpr1_ci"]
                    ax.bar([x], [s[metric]], w, yerr=[[s[metric] - c[0]], [c[1] - s[metric]]], color=COLORS[cond], hatch=hatch,
                           label=COND_LABEL[cond], capsize=2, error_kw=dict(lw=0.8))
            ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
            ax.set_ylabel(lab); ax.set_ylim(0, 1.05)
            if metric == "auroc":
                ax.axhline(0.5, ls=":", color="k", lw=0.8)
            ax.set_title(f"{JUDGE_LABEL.get(judge, judge)}: {lab}")
        h, l = axes[1].get_legend_handles_labels()
        seen = {}
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
        axes[1].legend(seen.values(), seen.keys(), fontsize=8, loc="upper right", ncol=2)
        fig.tight_layout()
        out = os.path.join(args.img_dir, f"mask_judge_{args.tag}{VARIANT}_{judge}.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print("figure", out)
    json.dump(summary, open(args.json_out, "w"), indent=1)
    if args.examples_md:
        # heatmap of the per-example scores
        for judge in args.judges:
            L = load(args.tag, judge, args.judge_dir)
            if L is None:
                continue
            rows, key = L
            conds = [c for c in CONDS if (args.examples_case, c) in rows]
            tags = sorted({t for c in conds for t in rows[(args.examples_case, c)]})
            if not tags or not conds:
                continue
            M = np.full((len(tags), len(conds)), np.nan)
            for ti, t in enumerate(tags):
                for ci, c in enumerate(conds):
                    v = rows[(args.examples_case, c)].get(t)
                    if v is not None:
                        M[ti, ci] = v[1] * (100 if key == "p_yes" else 1)
            fig, ax = plt.subplots(figsize=(1.2 * len(conds) + 3, 0.28 * len(tags) + 1.5))
            im = ax.imshow(M, vmin=0, vmax=100, cmap="Reds", aspect="auto")
            ax.set_xticks(range(len(conds))); ax.set_xticklabels([COND_LABEL.get(c, c) for c in conds], rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(tags))); ax.set_yticklabels([t.split("_", 2)[-1] for t in tags], fontsize=7)
            ax.set_title(f"{JUDGE_LABEL.get(judge, judge)}: {args.examples_case} class, judge probability per example", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.03)
            fig.tight_layout(); fig.savefig(os.path.join(args.img_dir, f"examples_{args.examples_case}_{args.tag}{VARIANT}_{judge}.png"), dpi=120); plt.close(fig)
        ex = []
        for judge in args.judges:
            L = load(args.tag, judge, args.judge_dir)
            if L is None:
                continue
            rows, key = L
            conds = [c for c in CONDS if (args.examples_case, c) in rows]
            tags = sorted({t for c in conds for t in rows[(args.examples_case, c)]})
            ex.append(f"\n**{JUDGE_LABEL.get(judge, judge)}**, class {args.examples_case}: score per example and condition "
                      f"({'P(yes)' if key == 'p_yes' else '0-100'}; a positive is right when >= {'0.5' if key == 'p_yes' else '50'}, a negative when below)\n")
            ex.append("| example | label | " + " | ".join(COND_LABEL.get(c, c) for c in conds) + " |")
            ex.append("|---|---|" + "---|" * len(conds))
            for t in tags:
                lab = None
                cells = []
                for c in conds:
                    v = rows[(args.examples_case, c)].get(t)
                    if v is not None:
                        lab = "pos" if v[0] else "neg"
                        cells.append(f"{v[1]:.2f}" if key == "p_yes" else f"{v[1]:.0f}")
                    else:
                        cells.append("-")
                ex.append(f"| {t} | {lab} | " + " | ".join(cells) + " |")
        open(args.examples_md, "w").write("\n".join(ex))
    text = "\n".join(md)
    if args.md:
        open(args.md, "w").write(text)
    if args.notes:
        import datetime, glob
        done = {m: len(glob.glob(f"results/prompt_bias_masks/masks_{args.tag}/masks/ex*_{m}*.json")) for m in ("p2t_rg", "p2t_kl")}
        done["ta"] = len(glob.glob(f"results/prompt_bias_masks/masks_{args.tag}/masks_ta/ex*_thought_anchors.json"))
        done["attribution"] = len(glob.glob(f"results/prompt_bias_masks/masks_{args.tag}/edges/ex*.edges.json"))
        figs = "\n\n".join(f"![{JUDGE_LABEL.get(j, j)}](../images/prompt_bias_masks/mask_judge_{args.tag}{VARIANT}_{j}.png)" for j in args.judges
                            if os.path.exists(os.path.join(args.img_dir, f"mask_judge_{args.tag}{VARIANT}_{j}.png")))
        body = (f"*Status ({datetime.datetime.now():%Y-%m-%d %H:%M}): masks finished per method, out of 90 examples: "
                + ", ".join(f"{k} {v}" for k, v in done.items()) + f". {args.status}*\n\n"
                "*Table: per judge and class, the examples judged (positives/negatives), how many the judge gets right at a probability of 50 "
                "(positives at or above, negatives below), the AUROC with a bootstrap 95 percent interval and the TPR at 5 percent FPR (with fewer "
                "than 20 negatives this is the TPR at zero false positives; with 20, the threshold admits one negative). Rows marked *all examples* are the baseline and bias-informed-judge "
                "conditions on the whole class; rows marked *matched* restrict every condition to the examples judged under all four mask "
                "conditions, so the conditions of one class are compared on the same examples (AUROC and TPR are printed once the matched set has "
                f"at least {args.min_n} positives and {args.min_n} negatives). The selected set is the dataset as designed (5+5, 20+20, 20+20); "
                "the rows *with discrim_mp_explicit added* include the extension of 4+4 per class from the discrim-eval explicit family; *matched, selected set only* "
                "is the matched comparison without the extension, and it is what the figures show.*\n"
                + text + "\n\n" + figs + "\n\n*Figures: AUROC (left) and TPR at 5 percent FPR (right) per class and condition; bars are bootstrap 95 percent intervals; dotted: chance. Hatched bars: the full-set baseline and bias-informed-judge conditions of a class whose matched set is still too small; solid bars: every condition on the matched set.*\n")
        body = body.replace(text, "<details><summary>Table (click to expand)</summary>\n" + text + "\n</details>")
        for judge in args.judges:
            hm = os.path.join(args.img_dir, f"examples_{args.examples_case}_{args.tag}{VARIANT}_{judge}.png")
            if os.path.exists(hm):
                body += f"\n![per-example scores, {args.examples_case}, {judge}](../images/prompt_bias_masks/{os.path.basename(hm)})\n"
        body += (f"\n*Per-example scores of the {args.examples_case} class (rows: examples, columns: conditions, colour: the judge's probability; "
                 "positives should be dark, negatives light).*\n")
        n = open(args.notes).read()
        a, b = n.index(f"<!-- {args.marker}_START -->") + len(f"<!-- {args.marker}_START -->"), n.index(f"<!-- {args.marker}_END -->")
        open(args.notes, "w").write(n[:a] + "\n" + body + "\n" + n[b:])
        print("notes updated:", args.notes)
    print(text)


if __name__ == "__main__":
    main()
