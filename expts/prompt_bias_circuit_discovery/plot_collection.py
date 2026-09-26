"""Figure: how much each demographic cue moves the decision, per profile.

Reads the collection report (16 rollouts per prompt variant) and draws, for
each setting, a heatmap of P(A) of every cue variant minus P(A) of the
neutral variant of the same profile (A = approve the loan / invite the
candidate).
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "xtick.labelsize": 12, "ytick.labelsize": 13})

AXES = [("race", ["white", "black", "hispanic", "asian"]), ("name", ["white", "black", "hispanic", "asian"]),
        ("religion", ["christian", "muslim", "jewish", "hindu"]), ("gender", ["man", "woman"]),
        ("nationality", ["germany", "nigeria", "mexico", "china"])]
NAMES = {"white": "Connor Walsh", "black": "DeShawn Washington", "hispanic": "Luis Hernandez", "asian": "Wei Zhang"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="results/prompt_bias/rollouts_report.json")
    ap.add_argument("--out", default="notes/images/prompt_to_trace_bias/bias_fig0_collection.png")
    args = ap.parse_args()
    rep = json.load(open(args.report))["report"]
    by = {(r["setting"], r["profile_id"], r["axis"], r["value"]): r["p_A"] for r in rep}
    cols = [(a, v) for a, vs in AXES for v in vs]
    labels = [f"{a}: {NAMES[v] if a == 'name' else v}" for a, v in cols]
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    for ax, setting, title in [(axes[0], "hiring", "resume screening: P(invite) minus P(invite) of the neutral variant"),
                               (axes[1], "loan", "loan application: P(approve) minus P(approve) of the neutral variant")]:
        M = np.array([[by[(setting, p, a, v)] - by[(setting, p, "neutral", "none")] for a, v in cols] for p in range(10)])
        neutral = [by[(setting, p, "neutral", "none")] for p in range(10)]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(cols)))
        if setting == "loan":
            ax.set_xticklabels(labels, rotation=60, ha="right")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"profile {p} (neutral {neutral[p]:.2f})" for p in range(10)])
        for i in range(10):
            for j in range(len(cols)):
                if abs(M[i, j]) >= 0.25:
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=10,
                            color="white" if abs(M[i, j]) > 0.4 else "black")
        for b in [3.5, 7.5, 11.5, 13.5]:
            ax.axvline(b, color="k", lw=1)
        ax.set_title(title)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("change in P(A) relative to the neutral variant (16 rollouts each)")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("saved", args.out)
    # numbers for the text
    for setting in ["hiring", "loan"]:
        M = np.array([[by[(setting, p, a, v)] - by[(setting, p, "neutral", "none")] for a, v in cols] for p in range(10)])
        print(setting, "mean delta per axis:", {a: round(float(np.mean([M[:, j] for j, (aa, v) in enumerate(cols) if aa == a])), 3) for a, _ in AXES})


if __name__ == "__main__":
    main()
