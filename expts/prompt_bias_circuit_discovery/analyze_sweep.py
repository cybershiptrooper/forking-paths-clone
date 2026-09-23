"""Figures and summary numbers for one prompt-to-trace sweep.

Reads, under results/prompt_bias/<sweep>/:
  selection.json          the prompts (tag, target/biased letter, verbalizes_cue, ...)
  eval/*.eval.json        learned masks at their matched sparsity (rg = reward gap
                          toward the answer without the cue, kl = answer-KL preservation)
  eval_ta/*.eval.json     Thought Anchors scores at every sparsity
  random/*.random_eval.json  random masks over the same pool (3 draws per sparsity)
  edges/*.edges.json      column and single-cell ablations
  masks/*.json            the learned masks (which cells they keep)
and writes PNGs to notes/images/<image_dir>/ plus summary.json.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.analyze_sweep \
        --sweep p2t_hint --image_dir prompt_to_trace_bias --prefix hint \
        --cue_regex "professor|stanford"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13})
C_LEARN, C_TA, C_RAND, C_KL = "#2166ac", "#e08214", "#7f7f7f", "#1b7837"


def load_json(p):
    with open(p) as f:
        return json.load(f)


def letter_index(letters, letter):
    st = [l.strip() for l in letters]
    return st.index(letter.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--prefix", required=True, help="file-name prefix for the figures")
    ap.add_argument("--cue_regex", default=None,
                    help="Regex that identifies the cue sentence among the prompt "
                    "sentences (default: selection's cue_sentence_idx).")
    args = ap.parse_args()
    root = f"results/prompt_bias/{args.sweep}"
    img = f"notes/images/{args.image_dir}"
    os.makedirs(img, exist_ok=True)
    selected = load_json(f"{root}/selection.json")
    by_tag = {s["tag"]: s for s in selected}

    # ---------------- learned masks (matched sparsity) --------------------
    learned = {}  # (tag, obj, tsp) -> row
    for p in glob.glob(f"{root}/eval/*.eval.json"):
        e = load_json(p)
        stem = os.path.basename(p).replace(".eval.json", "")
        m = re.match(rf"{args.sweep}_(.+)_(rg|kl)_tsp(\d+)$", stem)
        if not m:
            continue
        tag, obj, tsp = m.group(1), m.group(2), int(m.group(3)) / 100
        row = next(r for r in e["rows"] if r["row"] == "mask_eval" and r["mode"] == "top_k")
        learned[(tag, obj, tsp)] = dict(row, kl_max=e["kl_max"], letters=e["answer_letters"],
                                        clean=e["clean_answer_probs"])
    # ---------------- TA ---------------------------------------------------
    ta = {}
    for p in glob.glob(f"{root}/eval_ta/*.eval.json"):
        e = load_json(p)
        stem = os.path.basename(p).replace("_thought_anchors.eval.json", "")
        tag = stem.replace(f"{args.sweep}_", "")
        for r in e["rows"]:
            if r["row"] == "mask_eval" and r["mode"] == "top_k":
                ta[(tag, r["target_sparsity"])] = dict(r, kl_max=e["kl_max"], letters=e["answer_letters"],
                                                       clean=e["clean_answer_probs"])
    # ---------------- random -----------------------------------------------
    rnd = {}
    for p in glob.glob(f"{root}/random/*.random_eval.json"):
        e = load_json(p)
        tag = os.path.basename(p).replace(".random_eval.json", "").replace(f"{args.sweep}_", "")
        for r in e.get("rows", []):
            if "target_sparsity" in r and "mean_answer_probs" in r:
                # one entry per random draw, each with its own answer distribution
                for kl_s, p_s in zip(r["sample_kls"], r["sample_answer_probs"]):
                    rnd.setdefault((tag, r["target_sparsity"]), []).append(
                        dict(kl=kl_s, answer_probs=p_s, kl_max=e["kl_max"], letters=e["answer_letters"]))
    # ---------------- edges -------------------------------------------------
    edges = {}
    for p in glob.glob(f"{root}/edges/*.edges.json"):
        tag = os.path.basename(p).replace(".edges.json", "").replace(f"{args.sweep}_", "")
        edges[tag] = load_json(p)

    tags = [s["tag"] for s in selected]
    print(f"{len(tags)} prompts; learned rows {len(learned)}, TA rows {len(ta)}, "
          f"random {len(rnd)}, edges {len(edges)}")

    def probs(row, tag, which):
        s = by_tag[tag]
        li = letter_index(row["letters"], s[which + "_letter"])
        return row["answer_probs"][li]

    def clean_probs(tag, which):
        for key in [(tag, "rg", 0.05)] + [(tag, sp) for sp in (0.05, 0.1)]:
            if key in learned:
                return probs(dict(learned[key], answer_probs=learned[key]["clean"]), tag, which)
            if key in ta:
                return probs(dict(ta[key], answer_probs=ta[key]["clean"]), tag, which)
        return np.nan

    summary = {"n_prompts": len(tags), "prompts": {}}
    # keep only the sparsities that at least half the prompts were trained at,
    # so the median curves are over the prompt set and not over a few prompts
    def _sps(obj):
        counts = {}
        for (t, o, sp) in learned:
            if o == obj:
                counts[sp] = counts.get(sp, 0) + 1
        return sorted(sp for sp, c in counts.items() if c >= len(tags) / 2)
    flip_sps = _sps("rg")
    kl_sps = _sps("kl")
    print("sparsities used: flip", flip_sps, "keep", kl_sps)
    ta_sps = sorted({k[1] for k in ta})

    # ================= Figure 1: flip curves =================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, which, title in [(axes[0], "biased", "P(answer of the stored trace)"),
                             (axes[1], "target", "P(answer without the cue)")]:
        def series(getter, sps):
            M = np.full((len(tags), len(sps)), np.nan)
            for ti, t in enumerate(tags):
                for si, sp in enumerate(sps):
                    v = getter(t, sp)
                    if v is not None:
                        M[ti, si] = v
            return M
        L = series(lambda t, sp: probs(learned[(t, "rg", sp)], t, which) if (t, "rg", sp) in learned else None, flip_sps)
        T = series(lambda t, sp: probs(ta[(t, sp)], t, which) if (t, sp) in ta else None, ta_sps)
        R = series(lambda t, sp: (np.mean([probs(r, t, which) for r in rnd[(t, sp)]]) if (t, sp) in rnd else None), ta_sps)
        for ti in range(len(tags)):
            ax.plot(np.array(flip_sps) * 100, L[ti], color=C_LEARN, alpha=0.18, lw=1)
        cl = np.array([clean_probs(t, which) for t in tags])
        ax.axhline(np.nanmedian(cl), color="k", ls=":", lw=1.5, label="no ablation (median)")
        ax.plot(np.array(flip_sps) * 100, np.nanmedian(L, 0), "o-", color=C_LEARN, lw=2.5, ms=8,
                label="learned mask (flip objective)")
        ax.plot(np.array(ta_sps) * 100, np.nanmedian(T, 0), "s-", color=C_TA, lw=2.5, ms=8, label="Thought Anchors")
        ax.plot(np.array(ta_sps) * 100, np.nanmedian(R, 0), "^-", color=C_RAND, lw=2.5, ms=8, label="random mask")
        ax.set_xlabel("prompt-to-trace cells removed (%)")
        ax.set_ylabel(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        summary[f"median_p_{which}"] = {"learned": dict(zip(map(str, flip_sps), np.nanmedian(L, 0).tolist())),
                                        "ta": dict(zip(map(str, ta_sps), np.nanmedian(T, 0).tolist())),
                                        "random": dict(zip(map(str, ta_sps), np.nanmedian(R, 0).tolist())),
                                        "clean_median": float(np.nanmedian(cl))}
        if which == "target":
            # min ablation to flip (target becomes argmax)
            flips = {}
            for ti, t in enumerate(tags):
                fl = None
                for si, sp in enumerate(flip_sps):
                    k = (t, "rg", sp)
                    if k in learned:
                        ap_ = learned[k]["answer_probs"]
                        if int(np.argmax(ap_)) == letter_index(learned[k]["letters"], by_tag[t]["target_letter"]):
                            fl = sp
                            break
                flips[t] = fl
            summary["min_sparsity_to_flip"] = flips
    axes[0].legend(loc="lower left")
    fig.suptitle(f"{args.prefix}: masks over prompt-to-trace cells only, n = {len(tags)} prompts, median over prompts")
    fig.tight_layout()
    fig.savefig(f"{img}/{args.prefix}_fig1_flip_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= Figure 2: silent vs verbalized =========================
    groups = {"trace does not mention the cue": [t for t in tags if not by_tag[t].get("verbalizes_cue")],
              "trace mentions the cue": [t for t in tags if by_tag[t].get("verbalizes_cue")]}
    if all(len(v) for v in groups.values()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for ax, which, title in [(axes[0], "biased", "P(answer of the stored trace)"),
                                 (axes[1], "target", "P(answer without the cue)")]:
            for (gname, gt), col, mk in zip(groups.items(), [C_LEARN, C_KL], ["o", "D"]):
                M = np.array([[probs(learned[(t, "rg", sp)], t, which) if (t, "rg", sp) in learned else np.nan
                               for sp in flip_sps] for t in gt])
                for row in M:
                    ax.plot(np.array(flip_sps) * 100, row, color=col, alpha=0.2, lw=1)
                ax.plot(np.array(flip_sps) * 100, np.nanmedian(M, 0), mk + "-", color=col, lw=2.5, ms=8,
                        label=f"{gname} (n={len(gt)})")
            ax.set_xlabel("prompt-to-trace cells removed (%)")
            ax.set_ylabel(title)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
        axes[0].legend(loc="lower left")
        fig.suptitle(f"{args.prefix}: learned flip mask, split by whether the reasoning mentions the cue")
        fig.tight_layout()
        fig.savefig(f"{img}/{args.prefix}_fig2_silent_vs_verbal.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        summary["groups"] = {k: v for k, v in groups.items()}

    # ================= Figure 3: KL preservation ==============================
    if kl_sps:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        def klnorm(row):
            # raw KL in nats: the whole-pool ceiling (kl_max) is tiny on these
            # prompts, so a ratio to it is not informative
            return max(row["kl"], 1e-6)
        L = np.array([[klnorm(learned[(t, "kl", sp)]) if (t, "kl", sp) in learned else np.nan for sp in kl_sps] for t in tags])
        T = np.array([[klnorm(ta[(t, sp)]) if (t, sp) in ta else np.nan for sp in ta_sps] for t in tags])
        R = np.array([[np.mean([klnorm(r) for r in rnd[(t, sp)]]) if (t, sp) in rnd else np.nan for sp in ta_sps] for t in tags])
        for row in L:
            ax.plot(np.array(kl_sps) * 100, row, color=C_KL, alpha=0.18, lw=1)
        ax.plot(np.array(kl_sps) * 100, np.nanmedian(L, 0), "o-", color=C_KL, lw=2.5, ms=8, label="learned mask (KL objective)")
        ax.plot(np.array(ta_sps) * 100, np.nanmedian(T, 0), "s-", color=C_TA, lw=2.5, ms=8, label="Thought Anchors")
        ax.plot(np.array(ta_sps) * 100, np.nanmedian(R, 0), "^-", color=C_RAND, lw=2.5, ms=8, label="random mask")
        ax.set_xlabel("prompt-to-trace cells removed (%)")
        ax.set_ylabel("KL over the answer letters (nats)")
        allpool = [learned[(t, "kl", kl_sps[0])]["kl_max"] for t in tags if (t, "kl", kl_sps[0]) in learned]
        if allpool:
            ax.axhline(max(np.median(allpool), 1e-6), color="k", ls=":", lw=1.5, label="every cell removed (median)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        ax.set_title(f"{args.prefix}: keeping the answer distribution\n(median over {len(tags)} prompts; exact zeros drawn at 1e-6)", fontsize=14)
        fig.tight_layout()
        fig.savefig(f"{img}/{args.prefix}_fig3_kl_preservation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        summary["kl_raw_median"] = {"learned": dict(zip(map(str, kl_sps), np.nanmedian(L, 0).tolist())),
                                     "ta": dict(zip(map(str, ta_sps), np.nanmedian(T, 0).tolist())),
                                     "random": dict(zip(map(str, ta_sps), np.nanmedian(R, 0).tolist()))}
        summary["kl_max_per_prompt"] = {t: learned[(t, "kl", kl_sps[0])]["kl_max"] for t in tags if (t, "kl", kl_sps[0]) in learned}

    # ================= cue column: ablation + mask usage ======================
    def cue_idx(tag):
        if not args.cue_regex:
            return by_tag[tag].get("cue_sentence_idx")
        e = edges.get(tag)
        if e is None:
            return None
        if args.cue_regex:
            rx = re.compile(args.cue_regex, re.IGNORECASE)
            for s in e["sentences"][:e["num_prompt_sentences"]]:
                if rx.search(s["text"]):
                    return s["idx"]
            return None
        return by_tag[tag].get("cue_sentence_idx")

    col_rows = []
    for t in tags:
        e = edges.get(t)
        ci = cue_idx(t)
        if e is None or ci is None:
            continue
        clean_pt = e["clean"]["p_target"]
        cols = e["columns"]
        d_abl = np.array([c["column_ablation"]["p_target"] - clean_pt for c in cols])
        kl_abl = np.array([c["column_ablation"]["kl"] for c in cols])
        d_only = np.array([c["column_only"]["p_target"] - clean_pt for c in cols])
        rank_abl = int((d_abl > d_abl[ci]).sum()) + 1  # 1 = largest rise of P(target)
        rank_kl = int((kl_abl > kl_abl[ci]).sum()) + 1
        col_rows.append(dict(tag=t, cue_idx=ci, n_prompt=e["num_prompt_sentences"], clean_p_target=clean_pt,
                             all_pool_p_target=e["all_pool_ablated"]["p_target"], all_pool_kl=e["all_pool_ablated"]["kl"],
                             d_abl_cue=float(d_abl[ci]), d_abl_max=float(d_abl.max()), d_abl_argmax=int(d_abl.argmax()),
                             kl_abl_cue=float(kl_abl[ci]), kl_abl_max=float(kl_abl.max()), kl_abl_argmax=int(kl_abl.argmax()),
                             rank_cue_by_dp=rank_abl, rank_cue_by_kl=rank_kl,
                             d_only_cue=float(d_only[ci]), d_only_all=d_only.tolist(), d_abl_all=d_abl.tolist(),
                             kl_abl_all=kl_abl.tolist(),
                             sentences=[s["text"] for s in e["sentences"][:e["num_prompt_sentences"]]]))
    summary["column_ablation"] = col_rows

    if col_rows:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        # (a) per prompt: delta P(target) when the cue column is removed vs the best other column
        x = np.arange(len(col_rows))
        cue = [r["d_abl_cue"] for r in col_rows]
        best_other = [max(v for j, v in enumerate(r["d_abl_all"]) if j != r["cue_idx"]) for r in col_rows]
        allpool = [r["all_pool_p_target"] - r["clean_p_target"] for r in col_rows]
        axes[0].bar(x - 0.27, cue, 0.27, color=C_LEARN, label="cue sentence column removed")
        axes[0].bar(x, best_other, 0.27, color=C_TA, label="largest other column removed")
        axes[0].bar(x + 0.27, allpool, 0.27, color=C_RAND, label="every prompt-to-trace cell removed")
        axes[0].axhline(0, color="k", lw=1)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([r["tag"].replace(args.sweep + "_", "") for r in col_rows], rotation=90, fontsize=11)
        axes[0].set_ylabel("change in P(reference answer)")
        axes[0].legend(fontsize=11)
        axes[0].set_title("removing one prompt column at a time", fontsize=14)
        # (b) rank of the cue column among prompt columns
        ranks = [r["rank_cue_by_kl"] for r in col_rows]
        nps = [r["n_prompt"] for r in col_rows]
        axes[1].hist(ranks, bins=np.arange(0.5, max(nps) + 1.5, 1), color=C_LEARN, edgecolor="white")
        axes[1].set_xlabel("rank of the cue column among the prompt columns,\nby KL when the column is removed (1 = largest)")
        axes[1].set_ylabel("number of prompts")
        axes[1].set_title(f"cue column rank, {len(col_rows)} prompts")
        fig.tight_layout()
        fig.savefig(f"{img}/{args.prefix}_fig4_column_ablation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # mask usage per column at each sparsity: fraction of the cue column's cells removed vs other columns
    usage = {}
    for p in glob.glob(f"{root}/masks/*.json"):
        stem = os.path.basename(p).replace(".json", "")
        m = re.match(rf"{args.sweep}_(.+)_(rg|kl)_tsp(\d+)$", stem)
        if not m:
            continue
        tag, obj, tsp = m.group(1), m.group(2), int(m.group(3)) / 100
        ci = cue_idx(tag)
        if ci is None:
            continue
        nm = load_json(p)
        npmt = nm["metadata"]["num_prompt_sentences"]
        S = np.array(nm["scores"]) if not isinstance(nm["scores"][0][0], list) else None
        if S is None:
            continue
        # scores are learnable only inside the pool; frozen cells were filled with 1.0 by the sparse loader
        # use the log_alpha ranking from the eval instead: reconstruct top-k over pool cells
        pool = np.zeros_like(S, dtype=bool)
        pool[npmt:, :npmt] = True
        nvalid = pool.sum()
        n_keep = int(round((1 - tsp) * nvalid))
        flat = np.where(pool, S, -np.inf).ravel()
        keep = np.zeros_like(flat, dtype=bool)
        keep[np.argsort(-flat)[:n_keep]] = True
        keep = keep.reshape(S.shape)
        removed = pool & ~keep
        frac_col = removed[npmt:, :].sum(0) / max(1, S.shape[0] - npmt)
        usage[(tag, obj, tsp)] = dict(frac_removed_cue=float(frac_col[ci]),
                                      frac_removed_other=float(np.delete(frac_col[:npmt], ci).mean()),
                                      frac_removed_all=frac_col[:npmt].tolist(), cue_idx=ci)
    if usage:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        u_flip = sorted({k[2] for k in usage if k[1] == "rg"})
        u_kl = sorted({k[2] for k in usage if k[1] == "kl"})
        for ax, obj, sps, col, ttl in [(axes[0], "rg", u_flip, C_LEARN, "flip objective"),
                                       (axes[1], "kl", u_kl, C_KL, "KL objective")]:
            for t in tags:
                row = [usage[(t, obj, sp)]["frac_removed_cue"] if (t, obj, sp) in usage else np.nan for sp in sps]
                ax.plot(np.array(sps) * 100, row, color=col, alpha=0.18, lw=1)
            cue_m = [np.nanmedian([usage[(t, obj, sp)]["frac_removed_cue"] for t in tags if (t, obj, sp) in usage] or [np.nan]) for sp in sps]
            oth_m = [np.nanmedian([usage[(t, obj, sp)]["frac_removed_other"] for t in tags if (t, obj, sp) in usage] or [np.nan]) for sp in sps]
            ax.plot(np.array(sps) * 100, cue_m, "o-", color=col, lw=2.5, ms=8, label="cue sentence column")
            ax.plot(np.array(sps) * 100, oth_m, "s--", color=C_RAND, lw=2.5, ms=8, label="other prompt columns (mean)")
            ax.plot([0, 100], [0, 1], ":", color="k", lw=1, label="uniform removal")
            ax.set_xlabel("prompt-to-trace cells removed (%)")
            ax.set_ylabel("fraction of the column's cells removed")
            ax.set_title(f"learned mask, {ttl}")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{img}/{args.prefix}_fig5_mask_column_usage.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        summary["mask_column_usage"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in usage.items()}

    # ================= Figure 6: per-cell ablation heatmaps for example prompts ==
    ex = sorted(col_rows, key=lambda r: -abs(r["all_pool_p_target"] - r["clean_p_target"]))[:3]
    if ex:
        fig, axes = plt.subplots(1, len(ex), figsize=(6.2 * len(ex), 7.5), squeeze=False)
        for ax, r in zip(axes[0], ex):
            e = edges[r["tag"]]
            npmt, ns = e["num_prompt_sentences"], e["num_sentences"]
            H = np.full((ns - npmt, npmt), np.nan)
            for c in e.get("edges", []):
                H[c["i"] - npmt, c["j"]] = c["p_target"] - e["clean"]["p_target"]
            v = max(0.02, float(np.nanmax(np.abs(H))))
            im = ax.imshow(H, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
            ax.axvline(r["cue_idx"] - 0.5, color="k", lw=2); ax.axvline(r["cue_idx"] + 0.5, color="k", lw=2)
            ax.set_xlabel("prompt sentence (key); cue sentence boxed")
            ax.set_ylabel("reasoning sentence (query), in trace order")
            ax.set_xticks(range(npmt))
            ax.set_title(f"{r['tag'].replace(args.sweep + '_', '')}\nclean P(ref)={r['clean_p_target']:.2f}, "
                         f"all reads removed {r['all_pool_p_target']:.2f}", fontsize=13)
            # overlay: cells the learned flip mask removes at 30 percent, if that mask exists
            mp = f"{root}/masks/{args.sweep}_{r['tag']}_rg_tsp30.json"
            if os.path.exists(mp):
                nm = load_json(mp); S = np.array(nm["scores"])
                pool = np.zeros_like(S, dtype=bool); pool[npmt:, :npmt] = True
                n_keep = int(round(0.7 * pool.sum()))
                flat = np.where(pool, S, -np.inf).ravel(); keep = np.zeros_like(flat, dtype=bool)
                keep[np.argsort(-flat)[:n_keep]] = True; removed = pool & ~keep.reshape(S.shape)
                ys, xs = np.nonzero(removed[npmt:, :npmt])
                ax.scatter(xs, ys, marker="x", s=28, color="k", linewidths=1.0, label="removed by flip mask (30%)")
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("change in P(reference answer)\nwhen this one cell is removed", fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{img}/{args.prefix}_fig6_edge_heatmaps.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ================= paired tests learned vs TA / random at shared sparsities ==
    tests = {}
    for sp in flip_sps:
        if sp not in ta_sps:
            continue
        a = [probs(learned[(t, "rg", sp)], t, "target") for t in tags if (t, "rg", sp) in learned and (t, sp) in ta]
        b = [probs(ta[(t, sp)], t, "target") for t in tags if (t, "rg", sp) in learned and (t, sp) in ta]
        c = [np.mean([probs(r, t, "target") for r in rnd[(t, sp)]]) for t in tags if (t, "rg", sp) in learned and (t, sp) in rnd]
        if len(a) >= 5:
            d = np.array(a) - np.array(b)
            tests[str(sp)] = dict(n=len(a), wins_vs_ta=int((d > 0).sum()), median_delta_vs_ta=float(np.median(d)),
                                  p_vs_ta=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
            if len(c) == len(a):
                d2 = np.array(a) - np.array(c)
                tests[str(sp)].update(wins_vs_random=int((d2 > 0).sum()), median_delta_vs_random=float(np.median(d2)),
                                      p_vs_random=float(wilcoxon(d2).pvalue) if np.any(d2 != 0) else 1.0)
    summary["paired_tests_p_target"] = tests
    for t in tags:
        summary["prompts"][t] = {k: by_tag[t].get(k) for k in ("verbalizes_cue", "target_letter", "biased_letter",
                                                              "analysis_sentence_step", "n_prompt_sentences",
                                                              "axis", "value", "setting", "delta_p_A")}
    with open(f"{img}/{args.prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k in ("median_p_target", "median_p_biased", "min_sparsity_to_flip", "paired_tests_p_target", "kl_norm_median")}, indent=1, default=float))
    print(f"figures -> {img}/{args.prefix}_fig*.png")


if __name__ == "__main__":
    main()
