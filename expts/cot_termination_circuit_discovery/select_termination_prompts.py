"""Select the (prompt, analysis step) pairs for the termination sweep.

Reads the scan JSONs, filters to pairs with a bimodal termination
fraction and enough terminated-and-correct samples to populate the
target cluster, keeps one analysis step per prompt (the one whose
termination fraction is closest to 0.5), and takes up to --max_prompts
pairs balanced across collection files.

Usage (CPU):
    uv run python -m expts.cot_termination_circuit_discovery.select_termination_prompts \
        --scan_dir results/cot_termination/scan \
        --output results/cot_termination/selection.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", required=True)
    ap.add_argument("--f_min", type=float, default=0.25)
    ap.add_argument("--f_max", type=float, default=0.75)
    ap.add_argument("--min_target_cluster", type=int, default=4,
                    help="Minimum samples that terminate with the trace's "
                    "own final answer (cluster 0).")
    ap.add_argument("--max_prefix_len", type=int, default=4500,
                    help="Skip pairs with longer prefixes: training cost "
                    "scales with prefix length x bank size x steps.")
    ap.add_argument("--max_prompts", type=int, default=20)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    per_file = {}
    for path in sorted(glob.glob(os.path.join(args.scan_dir, "scan_*.json"))):
        with open(path) as f:
            scan = json.load(f)
        eligible = [
            r for r in scan["results"]
            if args.f_min <= r["f_terminated"] <= args.f_max
            and r["n_target_cluster"] >= args.min_target_cluster
            and r["prefix_len"] <= args.max_prefix_len
        ]
        # one step per prompt: termination fraction closest to 0.5
        best = {}
        for r in eligible:
            pi = r["prompt_index"]
            if pi not in best or abs(r["f_terminated"] - 0.5) < abs(
                best[pi]["f_terminated"] - 0.5
            ):
                best[pi] = r
        ranked = sorted(best.values(), key=lambda r: abs(r["f_terminated"] - 0.5))
        per_file[scan["data_path"]] = ranked
        print(f"{scan['data_path']}: {len(scan['results'])} pairs -> "
              f"{len(eligible)} eligible -> {len(ranked)} prompts")

    # round-robin across files for balance
    selected = []
    idx = 0
    while len(selected) < args.max_prompts:
        advanced = False
        for dp, ranked in per_file.items():
            if idx < len(ranked) and len(selected) < args.max_prompts:
                selected.append(ranked[idx])
                advanced = True
        if not advanced:
            break
        idx += 1

    slim = [
        {k: v for k, v in r.items() if k != "samples"} for r in selected
    ]
    for r in slim:
        print(f"  {os.path.basename(r['data_path'])} p{r['prompt_index']:03d} "
              f"step={r['analysis_sentence_step']} prefix={r['prefix_len']} "
              f"remain={r.get('remaining_tokens')} "
              f"f_term={r['f_terminated']:.2f} target={r['n_target_cluster']}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "selected": slim}, f, indent=2)
    print(f"Selected {len(slim)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
