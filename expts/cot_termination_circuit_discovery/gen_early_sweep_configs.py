"""Generate training configs for the early-analysis-point experiments.

Reads the bank manifest written by scan_early_analysis_points.py and emits
two config families under configs/sweeps/:

- ``early_gradcheck/``: 100-step gradient health check (log_every 1) of all
  six losses on the shortest-prefix bank of each dataset — the same
  protocol as the original gradient-check report, on the new regime.
- ``early_comparison/``: 500-step training runs of the four boundary-hazard
  losses plus the pairwise ranking fallback on every collected bank, at
  target sparsity 0.10 (the setting adopted from the 512-token-horizon
  loss comparison).

Boundary-hazard configs are only emitted for banks whose boundary metadata
exists (build_boundary_data.py fails on prompts with no discriminative
wrap-up marker; those prompts keep only the pairwise configs).

Usage:
    uv run python -m expts.cot_termination_circuit_discovery.gen_early_sweep_configs \
        --manifest results/cot_termination/early_2200/banks/manifest.json \
        --boundary_dir results/cot_termination/early_2200/boundary_data
"""

from __future__ import annotations

import argparse
import json
import os

CFG_ROOT = "expts/cot_termination_circuit_discovery/configs/sweeps"

# (config tag, objective name, needs boundary data, masking algorithm or None)
LOSSES = [
    ("pairwise", "candidate_pairwise_logistic", False, None),
    ("pairwise_length", "candidate_pairwise_logistic_length", False, None),
    ("stop_prob", "boundary_stop_prob", True,
     "nodewise_subnetwork_probing_boundary_hazard"),
    ("stop_prob_soft", "boundary_stop_prob_soft", True,
     "nodewise_subnetwork_probing_boundary_hazard_probe_weighted"),
    ("hazard_lift", "boundary_hazard_lift", True,
     "nodewise_subnetwork_probing_boundary_hazard"),
    ("explen_eligible", "boundary_expected_length_eligible", True,
     "nodewise_subnetwork_probing_boundary_hazard"),
]
GRADCHECK_LOSSES = [t for t, *_ in LOSSES]
COMPARISON_LOSSES = ["pairwise", "stop_prob", "stop_prob_soft",
                     "hazard_lift", "explen_eligible"]


def write_cfg(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def cfg_lines(entry, tag, objective, needs_bd, algorithm, boundary_dir,
              steps, log_every, tsp, out_dir, name):
    stem = os.path.splitext(os.path.basename(entry["bank_path"]))[0]
    lines = [
        "base_config: expts/cot_termination_circuit_discovery/configs/base.yaml",
        f"data_path: {entry['data_path']}",
        f"prompt_index: {entry['prompt_index']}",
        f"analysis_sentence_step: {entry['analysis_sentence_step']}",
        f"answer_bank_path: {entry['bank_path']}",
        f"objective: {objective}",
        f"target_sparsity: {tsp}",
        f"num_training_steps: {steps}",
        f"log_every: {log_every}",
        f"output_dir: {out_dir}",
        f"file_name: {name}",
    ]
    if needs_bd:
        lines.append(f"masking_algorithm: {algorithm}")
        lines.append(
            f"boundary_data_path: {os.path.join(boundary_dir, stem + '.json')}"
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--boundary_dir", required=True)
    ap.add_argument("--target_sparsity", type=float, default=0.10)
    ap.add_argument("--gradcheck_steps", type=int, default=100)
    ap.add_argument("--comparison_steps", type=int, default=500)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    loss_by_tag = {t: (o, nb, alg) for t, o, nb, alg in LOSSES}

    def has_bd(entry):
        stem = os.path.splitext(os.path.basename(entry["bank_path"]))[0]
        return os.path.exists(os.path.join(args.boundary_dir, stem + ".json"))

    # gradcheck: shortest-prefix bank per dataset THAT HAS boundary data
    by_ds = {}
    for e in manifest:
        ds = os.path.basename(e["data_path"]).split("_")[0].split(".")[0]
        by_ds.setdefault(ds, []).append(e)
    gc_entries = []
    for ds, entries in sorted(by_ds.items()):
        entries = sorted(entries, key=lambda e: e["prefix_len"])
        with_bd = [e for e in entries if has_bd(e)]
        gc_entries.append((with_bd or entries)[0])

    n_gc = 0
    for e in gc_entries:
        stem = os.path.splitext(os.path.basename(e["bank_path"]))[0]
        for tag in GRADCHECK_LOSSES:
            obj, nb, alg = loss_by_tag[tag]
            if nb and not has_bd(e):
                continue
            name = f"egc_{stem}_{tag}"
            write_cfg(
                os.path.join(CFG_ROOT, "early_gradcheck", name + ".yaml"),
                cfg_lines(e, tag, obj, nb, alg, args.boundary_dir,
                          args.gradcheck_steps, 1, args.target_sparsity,
                          "results/snp_sweep/early_2200/gradcheck/masks",
                          name),
            )
            n_gc += 1

    n_cmp = 0
    for e in manifest:
        stem = os.path.splitext(os.path.basename(e["bank_path"]))[0]
        for tag in COMPARISON_LOSSES:
            obj, nb, alg = loss_by_tag[tag]
            if nb and not has_bd(e):
                print(f"skip {stem} {tag}: no boundary data "
                      f"(no discriminative wrap-up marker)")
                continue
            tsp_tag = int(round(args.target_sparsity * 100))
            name = f"ec_{stem}_{tag}_tsp{tsp_tag:02d}"
            write_cfg(
                os.path.join(CFG_ROOT, "early_comparison", name + ".yaml"),
                cfg_lines(e, tag, obj, nb, alg, args.boundary_dir,
                          args.comparison_steps, 10, args.target_sparsity,
                          "results/snp_sweep/early_2200/masks", name),
            )
            n_cmp += 1
    print(f"wrote {n_gc} gradcheck configs "
          f"({[os.path.basename(e['bank_path']) for e in gc_entries]}) "
          f"and {n_cmp} comparison configs")


if __name__ == "__main__":
    main()
