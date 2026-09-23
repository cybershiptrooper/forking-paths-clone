"""Figures for the sanitised bias pipeline note.

1. bias_rate_by_family.png: fraction of variants (and questions) that show
   the cue effect, per family, Qwen3-8B vs Qwen3-32B, screen and
   confirmation stages.
2. delta_distributions.png: |delta| distributions per family and model.
3. judge_auroc.png: AUROC per case (prompt_change / same_prompt /
   explicit_control), condition (baseline / ceiling) and judge, per base
   model and family.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.plot_bias_v2 --out_dir notes/images/prompt_bias_v2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAM_ORDER = ["resume", "discrim_mp_explicit", "discrim_mp_implicit", "discrim_mp_implicit_pooled", "bbq",
             "karvonen_hiring", "blindspot_loan", "blindspot_admission"]
FAM_LABEL = {"resume": "resume (one sentence)", "discrim_mp_explicit": "discrim-eval explicit\n(race word)",
             "discrim_mp_implicit": "discrim-eval implicit\n(name)", "discrim_mp_implicit_pooled": "discrim-eval implicit\n(2 names pooled)",
             "bbq": "BBQ ambiguous\n(stereotyped >= 0.5)", "karvonen_hiring": "Karvonen hiring\n(name, Meta context)",
             "blindspot_loan": "blind-spot loan\n(name)", "blindspot_admission": "blind-spot admission\n(name)"}
COL = {"qwen3_8b": "#4C72B0", "qwen3_32b": "#DD8452"}


def load_rates(analysis_dir):
    """(model, stage) -> merged analysis over the original and the papers'
    families (bias_rates_<model>_screen.json and bias_rates_<model>_papers_screen.json)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(analysis_dir, "bias_rates_*.json"))):
        tag = os.path.basename(f)[len("bias_rates_"):-5]
        m = re.match(r"(qwen3_\d+b)(_papers)?_(screen|confirm)$", tag)
        if not m:
            continue
        d = json.load(open(f))
        key = (m.group(1), m.group(3))
        if key not in out:
            out[key] = dict(summary={}, rows=[], pooled_by_value=[], mcnemar_by_value=[])
        out[key]["summary"].update(d["summary"])
        out[key]["rows"] += d["rows"]
        out[key]["pooled_by_value"] += d.get("pooled_by_value", [])
        out[key]["mcnemar_by_value"] += d.get("mcnemar_by_value", [])
    return out


def fig_bias_rate(rates, out):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.2), sharey=False)
    for ax, stage in zip(axes, ["screen", "confirm"]):
        keys = [k for k in rates if k[1] == stage]
        if not keys:
            ax.set_title(f"{stage}: not run"); continue
        models = sorted({k[0] for k in keys})
        w = 0.8 / len(models)
        for mi, model in enumerate(models):
            s = rates[(model, stage)]["summary"]
            xs, ys, labels = [], [], []
            for fi, fam in enumerate(FAM_ORDER):
                if fam not in s:
                    continue
                v = s[fam]
                xs.append(fi + mi * w); ys.append(v["frac_screen"])
                labels.append(f"{v['n_screen']}/{v['n_variants']}\n(q<.05: {v['n_q_lt_alpha']})")
            ax.bar(xs, ys, width=w, color=COL.get(model, None), label=model)
            for x, y, l in zip(xs, ys, labels):
                ax.text(x, y + 0.005, l, ha="center", va="bottom", fontsize=7)
        ax.set_xticks([fi + w * (len(models) - 1) / 2 for fi in range(len(FAM_ORDER))])
        ax.set_xticklabels([FAM_LABEL[f] for f in FAM_ORDER], fontsize=7)
        if stage == "screen":
            ax.set_ylabel("fraction of all variants with the cue effect\n(|delta| >= 0.25 and Fisher p < 0.05)")
            ax.set_title("screen: every variant, 16 rollouts")
        else:
            ax.set_ylabel("fraction of screened variants that replicate\n(|delta| >= 0.25 and Fisher p < 0.05 on fresh rollouts)")
            ax.set_title("confirmation: screened variants only, 64 fresh-seed rollouts")
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(0.12, ax.get_ylim()[1]))
    fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)


def fig_delta(rates, out):
    stage = "confirm" if any(k[1] == "confirm" for k in rates) else "screen"
    keys = sorted(k for k in rates if k[1] == stage)
    fams = [f for f in FAM_ORDER if f != "discrim_mp_implicit_pooled"]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.4 * len(fams), 3.6))
    for ax, fam in zip(axes, fams):
        for model, st in keys:
            rows = [r for r in rates[(model, st)]["rows"] if r["family"] == fam and r["delta"] is not None]
            if not rows:
                continue
            d = np.array([r["delta"] if fam != "bbq" else r["p1"] for r in rows])
            ax.hist(d, bins=np.linspace(-1, 1, 41) if fam != "bbq" else np.linspace(0, 1, 21), alpha=0.5, color=COL.get(model), label=f"{model} (n={len(d)})")
        ax.set_title(FAM_LABEL[fam].replace("\n", " "), fontsize=9)
        ax.set_xlabel("P(pushed) - P(pushed | baseline)" if fam != "bbq" else "P(stereotyped person)", fontsize=8)
        ax.axvline(0, color="k", lw=0.5)
        ax.legend(fontsize=7)
    fig.suptitle(f"{stage} stage: distribution of the cue effect per variant", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)


def fig_family_effects(rates, out):
    """Population-level cue effects in the papers' terms: for each cue value,
    the mean over questions of P(pushed) - P(pushed | baseline), with the
    Wilcoxon p over questions; screen stage (every question)."""
    keys = sorted(k for k in rates if k[1] == "screen")
    if not keys:
        return
    fams = [f for f in FAM_ORDER if f not in ("discrim_mp_implicit_pooled", "bbq") and any(g["family"] == f for k in keys for g in rates[k]["pooled_by_value"])]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.8 * len(fams), 5.0), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        per_model = {}
        for model, st in keys:
            G = [g for g in rates[(model, st)]["pooled_by_value"] if g["family"] == fam]
            if fam == "resume":
                G = [g for g in G if g["value"].endswith("|hiring")]
            per_model[model] = {g["value"]: g for g in G}
        values = sorted({v for d in per_model.values() for v in d})
        models = [m for m, _ in keys if per_model.get(m)]
        h = 0.8 / max(1, len(models))
        xmax = 0.0
        for mi, model in enumerate(models):
            ys = [vi + mi * h for vi in range(len(values))]
            xs = [per_model[model][v]["mean_delta"] if v in per_model[model] else 0.0 for v in values]
            xmax = max(xmax, max(xs + [0.0]))
            ax.barh(ys, xs, height=h * 0.95, color=COL.get(model), label=model)
            for y, v in zip(ys, values):
                g = per_model[model].get(v)
                if g is None:
                    continue
                mark = "**" if g["wilcoxon_p"] < 0.001 else ("*" if g["wilcoxon_p"] < 0.05 else "")
                if mark:
                    ax.text(g["mean_delta"] + (0.002 if g["mean_delta"] >= 0 else -0.002), y, mark, va="center", ha="left" if g["mean_delta"] >= 0 else "right", fontsize=7)
        ax.set_yticks([vi + h * (len(models) - 1) / 2 for vi in range(len(values))])
        ax.set_yticklabels([v.split("|")[0].replace("_n0", " (name 1)").replace("_n1", " (name 2)") for v in values], fontsize=6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(FAM_LABEL[fam].replace("\n", " "), fontsize=8)
        ax.set_xlabel("mean over questions of\nP(pushed) - P(pushed | baseline)", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("family-level cue effects (screen, 16 rollouts per prompt); * Wilcoxon p < 0.05 over questions, ** p < 0.001", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)


def load_judge_rows(judge_dir):
    rows = []
    for f in glob.glob(os.path.join(judge_dir, "*.json")):
        b = os.path.basename(f)
        if "cache" in b or b.startswith("test_"):
            continue
        d = json.load(open(f))
        m = re.match(r"(qwen3_\d+b)_(.+?)_(prompt_change_same_answer|prompt_change|same_prompt|explicit_control)_(.+)\.json", b)
        if not m:
            continue
        model, fam, case, judge = m.groups()
        judge = judge[:-len("_noreason")] if judge.endswith("_noreason") else judge
        has_pyes = any(k.endswith("|p_yes") for k in d["summary"])
        from expts.prompt_bias_circuit_discovery.judge_bias_v2 import tpr_at_fpr
        for k, v in d["summary"].items():
            case2, cond, key = k.split("|")
            if key != ("p_yes" if has_pyes else "probability"):
                continue
            its = [i for i in d["items"] if i["case"] == case2 and i["condition"] == cond and i.get(key) is not None]
            t1, tlo, thi = tpr_at_fpr([i[key] for i in its if i["is_positive"]], [i[key] for i in its if not i["is_positive"]], 0.01)
            rows.append(dict(model=model, family=fam, case=case2, cond=cond, judge=judge, auroc=v["auroc"], ci=v["ci"], n=(v["n_pos"], v["n_neg"]),
                             tpr1=t1, tpr1_ci=[tlo, thi]))
    return rows


JUDGE_LABEL = {"gpt56luna": "GPT-5.6 Luna", "gemini38flash": "Gemini 3.8 Flash", "qwen32b": "Qwen3-32B (P(yes))"}
CASE_TITLE = {"prompt_change": "prompt_change: cue variant with the pushed answer vs an attribute-bearing baseline with the reference answer",
              "prompt_change_same_answer": "prompt_change_same_answer: cue variant vs baseline, both with the pushed answer (answer polarity held fixed)",
              "same_prompt": "same_prompt: the cue variant's pushed-answer vs reference-answer traces",
              "explicit_control": "explicit_control: the pushed-answer traces with one inserted sentence naming the attribute as the reason, vs the same_prompt negatives"}


def fig_judge(judge_dir, out_dir):
    """One figure per case and metric (AUROC; TPR at 1 percent FPR): rows =
    base model of the traces, columns = family, x = condition (baseline,
    ceiling, no reasoning), one point per judge."""
    rows = load_judge_rows(judge_dir)
    if not rows:
        return
    _fig_judge_metric(rows, out_dir, "auroc", "ci", "AUROC", "judge_auroc", 0.5)
    _fig_judge_metric(rows, out_dir, "tpr1", "tpr1_ci", "TPR at 1% FPR", "judge_tpr1fpr", None)
    _fig_increment(rows, out_dir)


def _fig_judge_metric(rows, out_dir, key, cikey, ylabel, prefix, chance):
    models = sorted({r["model"] for r in rows})
    judges = [j for j in JUDGE_LABEL if any(r["judge"] == j for r in rows)]
    jcol = dict(zip(judges, ["#DD8452", "#4C72B0", "#55A868"]))
    conds = ["baseline", "ceiling", "no_reasoning"]
    for case in ["prompt_change", "prompt_change_same_answer", "same_prompt", "explicit_control"]:
        R = [r for r in rows if r["case"] == case]
        if not R:
            continue
        fams = [f for f in FAM_ORDER if any(r["family"] == f for r in R)]
        fig, axes = plt.subplots(len(models), len(fams), figsize=(2.6 * len(fams), 3.0 * len(models)), squeeze=False, sharey=True)
        for mi, model in enumerate(models):
            for fi, fam in enumerate(fams):
                ax = axes[mi][fi]
                for ci, cond in enumerate(conds):
                    for ji, judge in enumerate(judges):
                        r = next((z for z in R if z["model"] == model and z["family"] == fam and z["cond"] == cond and z["judge"] == judge), None)
                        if r is None or r[key] is None:
                            continue
                        lo, hi = r[cikey]
                        ax.errorbar(ci + (ji - 1) * 0.22, r[key], yerr=[[r[key] - lo], [hi - r[key]]], fmt="o", color=jcol[judge], ms=4, capsize=2,
                                    label=JUDGE_LABEL[judge] if (mi == 0 and fi == 0 and ci == 0) else None)
                n = next((z["n"] for z in R if z["model"] == model and z["family"] == fam), None)
                if chance is not None:
                    ax.axhline(chance, color="k", ls=":", lw=0.8)
                ax.set_ylim(0, 1.05); ax.set_xticks(range(len(conds))); ax.set_xticklabels(["baseline", "ceiling", "no reasoning"], fontsize=7)
                ax.set_title(f"{FAM_LABEL[fam].replace(chr(10), ' ')}\n{model} traces" + (f", n = {n[0]}/{n[1]}" if n else ""), fontsize=7.5)
                if fi == 0:
                    ax.set_ylabel(ylabel)
                if mi == 0 and fi == 0:
                    ax.legend(fontsize=6.5, loc="lower left")
        note = "" if key == "auroc" else " (with fewer than 100 negatives the threshold sits above every negative: TPR at 0 false positives)"
        fig.suptitle(CASE_TITLE[case] + note, fontsize=8.5)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_{case}.png"), dpi=150); plt.close(fig)


def _fig_increment(rows, out_dir):
    models = sorted({r["model"] for r in rows})
    judges = [j for j in JUDGE_LABEL if any(r["judge"] == j for r in rows)]
    jcol = dict(zip(judges, ["#DD8452", "#4C72B0", "#55A868"]))
    # reasoning increment: baseline minus no_reasoning, per family / case / judge / model
    fig, axes = plt.subplots(1, len(models), figsize=(8.5 * len(models), 5.2), squeeze=False)
    cases = ["prompt_change", "prompt_change_same_answer", "same_prompt"]
    for mi, model in enumerate(models):
        ax = axes[0][mi]
        fams = [f for f in FAM_ORDER if any(r["family"] == f and r["model"] == model for r in rows)]
        labels, y = [], 0
        for fam in fams:
            for case in cases:
                pts = []
                for ji, judge in enumerate(judges):
                    b = next((z for z in rows if z["model"] == model and z["family"] == fam and z["case"] == case and z["cond"] == "baseline" and z["judge"] == judge), None)
                    nr = next((z for z in rows if z["model"] == model and z["family"] == fam and z["case"] == case and z["cond"] == "no_reasoning" and z["judge"] == judge), None)
                    if b and nr and b["auroc"] is not None and nr["auroc"] is not None:
                        ax.plot([nr["auroc"], b["auroc"]], [y + (ji - 1) * 0.22] * 2, "-", color=jcol[judge], lw=1)
                        ax.plot(b["auroc"], y + (ji - 1) * 0.22, "o", color=jcol[judge], ms=4, label=JUDGE_LABEL[judge] if not labels and ji < 3 and y == 0 else None)
                        ax.plot(nr["auroc"], y + (ji - 1) * 0.22, "o", mfc="white", color=jcol[judge], ms=4)
                        pts.append(1)
                if pts:
                    labels.append(f"{FAM_LABEL[fam].replace(chr(10), ' ')} | {case}")
                    y += 1
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6.5)
        ax.axvline(0.5, color="k", ls=":", lw=0.8); ax.set_xlim(0, 1.05)
        ax.set_xlabel("AUROC: open = reasoning withheld (question + answer only), filled = with the reasoning", fontsize=8)
        ax.set_title(f"{model} traces: what the judge gets from the chain of thought", fontsize=9)
        ax.invert_yaxis()
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "judge_reasoning_increment.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", default="results/prompt_bias_v2/analysis")
    ap.add_argument("--judge_dir", default="results/prompt_bias_v2/judge")
    ap.add_argument("--out_dir", default="notes/images/prompt_bias_v2")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rates = load_rates(args.analysis_dir)
    if rates:
        fig_bias_rate(rates, os.path.join(args.out_dir, "bias_rate_by_family.png"))
        fig_delta(rates, os.path.join(args.out_dir, "delta_distributions.png"))
        fig_family_effects(rates, os.path.join(args.out_dir, "family_effects.png"))
    fig_judge(args.judge_dir, args.out_dir)
    old = os.path.join(args.out_dir, "judge_auroc.png")
    if os.path.exists(old):
        os.remove(old)
    print("figures in", args.out_dir, sorted(os.listdir(args.out_dir)))


if __name__ == "__main__":
    main()
