"""Compare a directory of NodeMasks vs the two suppression baselines.

For each mask:
- Threshold at multiple sparsities
- Install hooks, run forward on prefix+suffix, read P(A/B/C/D) and probe-KL
- Aggregate into a long-form CSV-like dict
- Save plots: per-letter probability vs sparsity per mask, plus a
  KL-vs-sparsity plot grouping baselines and sweep configs.

Usage: see ``--help`` or directly:

    uv run python -m expts.direct_answer_circuit_discovery.compare_sweep \\
        --baselines results/.../thought_anchors.json \\
                    results/.../suppress_on_answer.json \\
        --sweep_dirs results/.../sweeps/snp_gpqa32_s50/ \\
        --model_name Qwen/Qwen3-8B \\
        --data_path data/.../gpqa_filtered.json \\
        --prompt_index 32 \\
        --analysis_sentence_step 50 \\
        --sentence_gap 4 \\
        --sparsities 0.0 0.3 0.5 0.7 0.9 \\
        --output_dir notes/images/direct_answer_circuit_discovery \\
        --tag kl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import set_seed, clear_cuda
from utils.masks import (
    NodeMask,
    build_gap_filter,
    build_mode_filter,
    build_causal_filter,
    build_combined_filter,
)
from utils.circuit_eval import (
    build_token_to_sent_map,
    build_binary_masks,
    install_mask_hooks,
    remove_handles,
)
import numpy as np

from expts.direct_answer_circuit_discovery.probe import (
    DEFAULT_ANSWER_LETTERS,
    DEFAULT_SUFFIX,
    build_answer_probe,
)
from expts.direct_answer_circuit_discovery.learn import _build_prefix
from expts.direct_answer_circuit_discovery.compare_masks import (
    _threshold_for_sparsity,
    _kl_on_answer,
)


def _short_label(path: str) -> str:
    base = os.path.basename(path).removesuffix(".json")
    base = base.replace("qwen3_8b_gpqa32_at_s50_", "")
    base = base.replace("qwen3_8b_gpqa32_at_s50", "snp_default")
    return base


def main(
    *,
    baselines: List[str],
    sweep_dirs: List[str],
    model_name: str,
    data_path: Optional[str] = None,
    prompt_index: Optional[int] = None,
    prompt: Optional[str] = None,
    base_answer_type: str = "stored",
    analysis_timestep: Optional[int] = None,
    analysis_sentence_step: Optional[int] = None,
    sentences_after_prefix: int = 0,
    probe_suffix: str = DEFAULT_SUFFIX,
    answer_letters: Optional[List[str]] = None,
    sentence_gap: int = 4,
    sentence_chunk: int = 1,
    min_sentence_length: int = 10,
    sparsities: Optional[List[float]] = None,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "notes/images/direct_answer_circuit_discovery",
    tag: str = "comparison",
):
    if answer_letters is None:
        answer_letters = list(DEFAULT_ANSWER_LETTERS)
    if sparsities is None:
        sparsities = [0.0, 0.3, 0.5, 0.7, 0.9]

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    masks: List[str] = []
    for b in baselines:
        if os.path.exists(b):
            masks.append(b)
        else:
            print(f"WARNING: baseline not found: {b}")
    for d in sweep_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            masks.append(f)
    print(f"Total masks to evaluate: {len(masks)}")

    print("Building prefix + probe...")
    tok = AutoTokenizer.from_pretrained(model_name)
    prefix_ids, sentences, prompt, correct_answer, _, _ = _build_prefix(
        tokenizer=tok,
        prompt=prompt,
        data_path=data_path,
        prompt_index=prompt_index,
        base_answer_type=base_answer_type,
        analysis_timestep=analysis_timestep,
        analysis_sentence_step=analysis_sentence_step,
        sentences_after_prefix=sentences_after_prefix,
        min_sentence_length=min_sentence_length,
        sentence_chunk=sentence_chunk,
    )
    prefix_len = prefix_ids.shape[-1]
    num_sents = len(sentences)
    probe = build_answer_probe(tok, suffix=probe_suffix, answer_letters=answer_letters)
    print(f"  prefix_len={prefix_len}, sentences={num_sents}, correct={correct_answer!r}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    target_device = next(model.parameters()).device
    input_ids = prefix_ids.to(target_device)
    cont = probe.make_continuation(target_device)
    full_input = torch.cat([input_ids, cont], dim=-1)
    answer_pos = probe.answer_logit_position(prefix_len)
    ans_ids = probe.answer_token_ids.to(target_device)

    model.eval()
    with torch.no_grad():
        clean_logits = model(full_input).logits
    clean_lp = F.log_softmax(
        clean_logits[0, answer_pos, ans_ids].float(), dim=-1,
    ).detach()
    clean_p = clean_lp.exp().cpu().tolist()
    del clean_logits
    torch.cuda.empty_cache()
    print(f"Clean P: {dict(zip(probe.answer_letters, clean_p))}")

    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    layers = list(range(num_layers))
    gap_filter = build_gap_filter(num_sents, sentence_gap, device=target_device)
    mode_filter = build_mode_filter(num_sents, num_sents, "prefix", device=target_device)
    causal_filter = build_causal_filter(num_sents, device=target_device)
    combined_filter = build_combined_filter(gap_filter, mode_filter, causal_filter)
    combined_filter_cpu = combined_filter.cpu()
    token_to_sent = build_token_to_sent_map(
        sentences, full_input.shape[-1], target_device,
    )

    def _install_explicit(scores_np):
        """Install hooks with an explicit (S, S) mask broadcast to all heads/layers.

        Used for SNP masks (continuous attenuation in [0,1]) so we apply the
        score matrix as the mask directly, bypassing percentile thresholding
        which is meaningless for bimodal SNP scores.
        """
        sc = torch.from_numpy(scores_np.astype(np.float32)).to(target_device)
        masks_t = sc.unsqueeze(0).expand(num_heads, -1, -1)
        bm = {l: masks_t for l in layers}
        return install_mask_hooks(
            model, layers, bm, token_to_sent, combined_filter, renormalize=True,
        )

    rows = []
    for mask_path in masks:
        nm = NodeMask.from_json(mask_path)
        algo = nm.algorithm
        label = _short_label(mask_path)
        is_snp = algo == "nodewise_subnetwork_probing_sdpa"
        print(f"\n[{label}] algorithm={algo}, granularity={nm.granularity}, is_snp={is_snp}")
        if set(nm.layers) != set(layers):
            missing = sorted(set(layers) - set(nm.layers))
            extra = sorted(set(nm.layers) - set(layers))
            raise ValueError(
                f"Mask {mask_path!r} was trained on layers {sorted(nm.layers)} "
                f"but eval applies hooks to {layers}. "
                f"Eval here always installs hooks on all model layers, so any "
                f"subset-of-layers mask would be evaluated against an "
                f"untouched model on the non-target layers — silently "
                f"different from the training-time non-target ablation. "
                f"Train with layers_to_analyse='all' or extend this eval to "
                f"replicate the training non-target treatment. "
                f"Missing from mask: {missing}; extra in mask: {extra}."
            )

        if is_snp:
            # SNP scores ARE the mask: edges with score==0 are pruned, edges
            # with score>0 are alive (with attenuation = score).  Per
            # NodewiseSubnetworkProbingSDPA._current_sparsity, sparsity is the
            # fraction of entries with HC mean exactly 0.
            scores_np = np.array(nm.scores)
            for variant, mode_label in [
                ("continuous", "cont"),
                ("binary_pos", "bin"),
            ]:
                if variant == "continuous":
                    eval_scores = scores_np
                else:
                    eval_scores = (scores_np > 0).astype(np.float32)
                handles = _install_explicit(eval_scores)
                try:
                    out = _kl_on_answer(model, full_input, answer_pos, ans_ids, clean_lp)
                finally:
                    remove_handles(handles)
                    torch.cuda.empty_cache()
                # Sparsity at the SNP-natural level (==0)
                snp_sp = float((scores_np == 0.0).sum() / scores_np.size)
                rows.append({
                    "mask_path": mask_path,
                    "label": f"{label}_{mode_label}",
                    "algorithm": algo,
                    "eval_mode": variant,
                    "target_sparsity": snp_sp,
                    "actual_sparsity": snp_sp,
                    "threshold": 0.0,
                    "probe_kl": out["kl"],
                    "p_masked": out["p_masked"],
                })
                p_str = " ".join(f"{l}={p:.3f}" for l, p in zip(probe.answer_letters, out["p_masked"]))
                print(f"  [{variant}] snp_sp={snp_sp:.3f} | KL={out['kl']:.4f} | {p_str}")
            continue

        # Continuous-score masks (thought_anchors, suppress-on-answer):
        # sweep percentile thresholds to get the sparsity-vs-KL curve.
        for sp in sparsities:
            thresh = _threshold_for_sparsity(nm, sp, combined_filter_cpu)
            binary_masks = build_binary_masks(
                nm, thresh, layers, num_heads, num_sents,
                combined_filter, target_device,
            )
            handles = install_mask_hooks(
                model, layers, binary_masks, token_to_sent,
                combined_filter, renormalize=True,
            )
            try:
                out = _kl_on_answer(model, full_input, answer_pos, ans_ids, clean_lp)
            finally:
                remove_handles(handles)
                del binary_masks
                torch.cuda.empty_cache()
            actual_sp = nm.sparsity(thresh, gap_filter=combined_filter_cpu)
            rows.append({
                "mask_path": mask_path,
                "label": label,
                "algorithm": algo,
                "eval_mode": "threshold",
                "target_sparsity": sp,
                "actual_sparsity": actual_sp,
                "threshold": thresh,
                "probe_kl": out["kl"],
                "p_masked": out["p_masked"],
            })
            p_str = " ".join(f"{l}={p:.3f}" for l, p in zip(probe.answer_letters, out["p_masked"]))
            print(f"  sp={sp:.2f} actual={actual_sp:.3f} | KL={out['kl']:.4f} | {p_str}")

    out_payload = {
        "tag": tag,
        "model_name": model_name,
        "prompt_index": prompt_index,
        "data_path": data_path,
        "prefix_len": prefix_len,
        "num_sentences": num_sents,
        "answer_letters": probe.answer_letters,
        "answer_token_ids": probe.answer_token_ids.tolist(),
        "answer_probs_clean": clean_p,
        "correct_letter_normalised": correct_answer,
        "sentence_gap": sentence_gap,
        "sparsities": sparsities,
        "rows": rows,
    }
    json_out = os.path.join(output_dir, f"{tag}_data.json")
    with open(json_out, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved data: {json_out}")

    # ---- Plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_label: dict = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    # 1) KL vs sparsity, all masks on one axis
    plt.figure(figsize=(8, 5))
    for label, xs in by_label.items():
        xs2 = sorted(xs, key=lambda r: r["actual_sparsity"])
        sps = [r["actual_sparsity"] for r in xs2]
        kls = [r["probe_kl"] for r in xs2]
        ls = "-" if "thought_anchors" in label or "suppress" in label or "attention" in label else "--"
        plt.plot(sps, kls, marker="o", linestyle=ls, label=label, linewidth=1.5, markersize=4)
    plt.xlabel("sparsity (fraction of edges dropped)")
    plt.ylabel("$\\mathrm{KL}(P_\\mathrm{clean} \\Vert P_\\mathrm{masked})$ over answer letters")
    plt.title(f"Probe-KL vs sparsity ({tag})")
    plt.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(output_dir, f"{tag}_kl_vs_sparsity.png")
    plt.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {p1}")

    # 2) Per-letter probability vs sparsity, one subplot per mask
    n_masks = len(by_label)
    cols = min(3, n_masks)
    rows_n = (n_masks + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.5 * cols, 3.2 * rows_n), squeeze=False)
    letters = probe.answer_letters
    color_map = {" A": "#1f77b4", " B": "#ff7f0e", " C": "#2ca02c", " D": "#d62728"}
    for ax, (label, xs) in zip(axes.flat, by_label.items()):
        xs2 = sorted(xs, key=lambda r: r["actual_sparsity"])
        sps = [r["actual_sparsity"] for r in xs2]
        for li, letter in enumerate(letters):
            vals = [r["p_masked"][li] for r in xs2]
            ax.plot(sps, vals, marker="o", color=color_map.get(letter, None),
                    label=letter.strip(), linewidth=1.5, markersize=4)
            ax.axhline(clean_p[li], color=color_map.get(letter, None),
                       linestyle=":", alpha=0.4, linewidth=1)
        ax.set_xlabel("sparsity")
        ax.set_ylabel("P(letter)")
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.flat[n_masks:]:
        ax.axis("off")
    fig.suptitle(f"P(letter) vs sparsity per mask ({tag}) — dotted = clean baseline")
    fig.tight_layout()
    p2 = os.path.join(output_dir, f"{tag}_pletter_vs_sparsity.png")
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {p2}")

    del model
    clear_cuda()
    return out_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", nargs="*", default=[])
    parser.add_argument("--sweep_dirs", nargs="*", default=[])
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--prompt_index", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--base_answer_type", default="stored")
    parser.add_argument("--analysis_timestep", type=int, default=None)
    parser.add_argument("--analysis_sentence_step", type=int, default=None)
    parser.add_argument("--sentences_after_prefix", type=int, default=0)
    parser.add_argument("--probe_suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--sentence_gap", type=int, default=4)
    parser.add_argument("--sentence_chunk", type=int, default=1)
    parser.add_argument("--sparsities", nargs="+", type=float,
        default=[0.0, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="notes/images/direct_answer_circuit_discovery")
    parser.add_argument("--tag", default="comparison")
    args = parser.parse_args()
    main(
        baselines=args.baselines,
        sweep_dirs=args.sweep_dirs,
        model_name=args.model_name,
        data_path=args.data_path,
        prompt_index=args.prompt_index,
        prompt=args.prompt,
        base_answer_type=args.base_answer_type,
        analysis_timestep=args.analysis_timestep,
        analysis_sentence_step=args.analysis_sentence_step,
        sentences_after_prefix=args.sentences_after_prefix,
        probe_suffix=args.probe_suffix,
        sentence_gap=args.sentence_gap,
        sentence_chunk=args.sentence_chunk,
        sparsities=args.sparsities,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        tag=args.tag,
    )
