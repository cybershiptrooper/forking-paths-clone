"""CLI for direct-answer circuit discovery.

Two modes selected by ``--mode``:

- ``learn``    — full circuit-discovery run (IG / activation patching /
                 etc.) using a probe objective.  Calls
                 :func:`learn.main`.
- ``suppress`` — attention-suppression run measuring KL on the answer
                 probe.  Calls :func:`suppress.main`.

Pass ``--config <path.yaml>`` to fill argument defaults from a YAML
file; CLI flags override config values.
"""

from __future__ import annotations

import argparse
from typing import List

from utils.expt_config import load_config


def _normalize_answer_letters(value) -> List[str]:
    """Accept a list (used as-is) or a comma-separated string.

    Comma-separated parsing strips one leading space per item (the
    natural ``", "`` separator) and then *prepends a single space* to
    every item.  This matches the convention required by the probe:
    answer letters must include a leading space so they share the
    suffix's tokenization (see ``probe.build_answer_probe``).  If you
    need a different formatting (e.g. no leading space, multi-character
    answer tokens), use the YAML list form like
    ``answer_letters: [" A", " B"]``.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    parts = [p for p in value.split(",") if p.strip()]
    return [" " + p.strip() for p in parts]


def _float_or_random(value):
    """--log_alpha_init accepts a float or the string 'random'."""
    if isinstance(value, str) and value.strip().lower() == "random":
        return "random"
    return float(value)


def _parse_layers(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if s.lower() == "all":
        return "all"
    return [int(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct-answer circuit discovery (suffix + answer probe)."
    )
    parser.add_argument(
        "--mode", choices=["learn", "suppress"], default="learn",
        help="learn = run a discovery algorithm; "
        "suppress = attention-suppression on the answer token.",
    )
    parser.add_argument("--config", type=str, default=None)

    # Shared
    parser.add_argument(
        "--model_name", type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    )
    parser.add_argument("--model_to_analyse", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--prompt_index", type=int, default=None)
    parser.add_argument("--correct_answer", type=str, default=None)
    parser.add_argument(
        "--base_answer_type",
        choices=["stored", "correct", "incorrect", "mode"],
        default="stored",
    )
    parser.add_argument("--analysis_timestep", type=int, default=None)
    parser.add_argument("--analysis_sentence_step", type=int, default=None)
    parser.add_argument(
        "--sentences_after_prefix", type=int, default=0,
        help="Append k extra reasoning sentences (taken from the model's "
        "stored base path) between the analysis-sentence prefix and the "
        "forced-answer suffix. Bridges fully-local probe (k=0) and a "
        "fully-resampled global outcome.",
    )
    parser.add_argument("--probe_suffix", type=str, default=None,
        help="Default: ' </think> I think the answer is '")
    parser.add_argument(
        "--answer_letters", type=str, default=None,
        help="Comma-separated list, e.g. 'A,B,C,D'.",
    )
    parser.add_argument("--sentence_gap", type=int, default=1)
    parser.add_argument("--sentence_chunk", type=int, default=1)
    parser.add_argument(
        "--mask_mode", choices=["prefix", "generation", "both"],
        default="prefix",
    )
    parser.add_argument(
        "--freeze_prompt_sentences", action="store_true", default=False,
        help="Freeze all attention to/from the prompt sentences (question, "
        "choices, chat template) at 1.0 so only reasoning-to-reasoning "
        "attention is learnable. Controls for the degenerate solution of "
        "ablating attention to the wrong answer options.",
    )
    parser.add_argument(
        "--no_renormalize_masked_attention",
        dest="renormalize_masked_attention",
        action="store_false",
    )
    parser.add_argument(
        "--backend", choices=["eager", "sdpa"], default="sdpa",
        help="Attention backend for score-generation forwards in TA / "
        "Suppress modes. Default sdpa to match SNP's training backend.",
    )
    parser.add_argument("--min_sentence_length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_dir",
        default="results/direct_answer_circuit_discovery",
    )
    parser.add_argument("--file_name", type=str, default=None)

    # learn-only
    parser.add_argument(
        "--masking_algorithm", type=str, default="nodewise_attribution",
    )
    parser.add_argument(
        "--objective",
        choices=[
            "answer_probe_kl",
            "answer_probe_reward_gap",
            "answer_probe_logit_margin",
            "answer_probe_prefix_kl",
            "answer_probe_target_kl",
            "candidate_reward_gap",
            "candidate_logprob_margin",
            "candidate_snis_reward_gap",
        ],
        default="answer_probe_kl",
    )
    parser.add_argument(
        "--target_probs", type=str, default=None,
        help="For answer_probe_target_kl: comma-separated target "
        "distribution over the answer letters (or a YAML list via config).",
    )
    parser.add_argument("--freeze_sentences_before", type=int, default=None)
    parser.add_argument(
        "--answer_bank_path", type=str, default=None,
        help="For candidate_* objectives (open-ended answers): path to the "
        "answer bank JSON built by build_answer_bank.py.",
    )
    parser.add_argument(
        "--target_letter", type=str, default=None,
        help="For answer_probe_reward_gap / answer_probe_logit_margin: "
        "which letter to promote. Defaults to dataset's correct_answer.",
    )
    parser.add_argument(
        "--logit_margin_reduce", choices=["mean", "max"], default="mean",
        help="For answer_probe_logit_margin: how to reduce other-letter "
        "logits before subtracting from the target logit.",
    )
    parser.add_argument(
        "--layers_to_analyse", type=str, default=None,
        help="Comma-separated layer indices, or 'all'.",
    )
    parser.add_argument(
        "--mask_granularity", choices=["head", "layer", "pair"],
        default="head",
    )
    parser.add_argument(
        "--pair_aggregation", choices=["sum", "mean", "median", "max"],
        default="mean",
    )
    parser.add_argument("--ablate_non_target_layers", action="store_true")
    parser.add_argument("--num_ig_steps", type=int, default=10)
    parser.add_argument("--no_negate_scores", action="store_true")
    parser.add_argument("--no_include_zero_ablation", action="store_true")
    parser.add_argument("--zero_ablation_epsilon", type=float, default=1e-10)
    parser.add_argument("--batch_chunk_size", type=int, default=None)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # Subnetwork-probing kwargs
    parser.add_argument("--l0_lambda", type=float, default=None)
    parser.add_argument("--num_training_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--log_alpha_init", type=_float_or_random, default=None,
        help="Initial gate location: a float (closed = -3, half = 0, "
             "open = +2) or 'random' for Uniform(-2, 2) per gate.",
    )
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--plot_every", type=int, default=None)
    parser.add_argument(
        "--sparsity_loss_mode",
        choices=["l0_mean", "target_size_relu"], default=None,
    )
    parser.add_argument(
        "--l0_normalize_hinge", action="store_true", default=None,
        help="Divide the target_size_relu hinge by the detached over-budget "
        "excess (floored at 1 cell) so total per-step sparsity pressure is "
        "bounded at ~lambda regardless of how far over budget the mask is.",
    )
    parser.add_argument(
        "--l0_warmup_frac", type=float, default=None,
        help="Fraction of training with lambda = 0 (default 0.25). Set 0 "
        "to apply sparsity pressure from step 0.",
    )
    parser.add_argument(
        "--l0_ramp_frac", type=float, default=None,
        help="Fraction of training over which lambda linearly ramps from 0 "
        "to l0_lambda (default 0.50).",
    )
    parser.add_argument("--target_sparsity", type=float, default=None)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd", "sgd_momentum", "hybrid"], default=None,
    )
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--l0_lr_multiplier", type=float, default=None)
    parser.add_argument("--dropout_p", type=float, default=None)
    parser.add_argument("--save_log_alpha", action="store_true", default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument(
        "--resume_from_checkpoint", action="store_true", default=None,
    )
    # D2 — HC variance reduction
    parser.add_argument("--num_hc_samples_per_step", type=int, default=None)
    parser.add_argument(
        "--log_alpha_init_mask_path", type=str, default=None,
        help="Mask JSON to initialize the gates from; mixed with the "
             "all-open mask by --log_alpha_init_mask_alpha.",
    )
    parser.add_argument("--log_alpha_init_mask_alpha", type=float, default=None)
    parser.add_argument("--polyak_ema_log_alpha", type=float, default=None)
    parser.add_argument("--hc_beta_anneal", action="store_true", default=None)
    parser.add_argument("--hc_beta_start", type=float, default=None)
    parser.add_argument("--hc_beta_end", type=float, default=None)
    parser.add_argument("--hc_beta_anneal_end_frac", type=float, default=None)
    # D3 — LR scheduler
    parser.add_argument(
        "--lr_schedule",
        choices=["constant", "cosine", "linear", "on_plateau"],
        default=None,
    )
    parser.add_argument("--lr_min_ratio", type=float, default=None)
    parser.add_argument("--lr_plateau_patience", type=int, default=None)
    parser.add_argument("--lr_plateau_factor", type=float, default=None)
    # DCM + PID (nodewise_dcm_pid_sdpa)
    parser.add_argument("--pid_kp", type=float, default=None)
    parser.add_argument("--pid_ki", type=float, default=None)
    parser.add_argument("--pid_kd", type=float, default=None)
    parser.add_argument("--pid_mult_init", type=float, default=None)
    parser.add_argument("--pid_ramp_end_frac", type=float, default=None)
    parser.add_argument("--pid_max_target_sparsity", type=float, default=None)
    parser.add_argument(
        "--snapshot_sparsities", type=str, default=None,
        help="Comma-separated sparsities at which the DCM+PID run snapshots "
        "its mask (YAML list form also accepted via --config).",
    )
    parser.add_argument("--dcm_mask_init", type=float, default=None)
    parser.add_argument(
        "--dcm_l0_optimizer", choices=["sgd", "adam"], default=None,
    )
    parser.add_argument("--dcm_polarization", type=float, default=None)
    parser.add_argument("--pid_snapshot_hold_steps", type=int, default=None)
    parser.add_argument("--dcm_lr_init", type=float, default=None)
    parser.add_argument("--dcm_lr_warmup_frac", type=float, default=None)
    return parser


def _shared_kwargs(args) -> dict:
    kwargs = dict(
        model_name=args.model_name,
        model_to_analyse=args.model_to_analyse,
        prompt=args.prompt,
        data_path=args.data_path,
        prompt_index=args.prompt_index,
        correct_answer=args.correct_answer,
        base_answer_type=args.base_answer_type,
        analysis_timestep=args.analysis_timestep,
        analysis_sentence_step=args.analysis_sentence_step,
        sentences_after_prefix=args.sentences_after_prefix,
        sentence_gap=args.sentence_gap,
        sentence_chunk=args.sentence_chunk,
        mask_mode=args.mask_mode,
        renormalize_masked_attention=args.renormalize_masked_attention,
        min_sentence_length=args.min_sentence_length,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        file_name=args.file_name,
    )
    if args.probe_suffix is not None:
        kwargs["probe_suffix"] = args.probe_suffix
    if args.answer_letters is not None:
        kwargs["answer_letters"] = _normalize_answer_letters(args.answer_letters)
    return kwargs


def main():
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if args.config:
        config = load_config(args.config)
        config.pop("config", None)
        parser.set_defaults(**config)
    args = parser.parse_args()

    # Opt-in bit-level reproducibility (config key: deterministic_algorithms).
    # Config keys land on args via set_defaults even without a parser arg.
    # Must run before the first CUDA op: cuBLAS reads the workspace config
    # at handle creation. Measured cost with allocation NaN-filling
    # switched off (see below): +15% per step for the sequential SNP
    # trainer / ~+1% for the HC-batched trainer on Qwen3-8B (hc = 4),
    # +0.4% batched on Qwen3-32B; the cost is the memory-efficient
    # attention backward serializing its split-key accumulation. Both
    # trainers rerun bit-identically under a fixed seed
    # (results/snp_sweep/hc_batched_validation/equivalence_summary.json,
    # results/snp_sweep/det_fill_off_check/summary.json).
    if getattr(args, "deterministic_algorithms", False):
        import os
        import torch
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        # Deterministic mode also turns on
        # torch.utils.deterministic.fill_uninitialized_memory, which
        # NaN-fills every freshly allocated tensor. That filling is a
        # debugging aid (it surfaces reads of uninitialized memory), not
        # part of the determinism guarantee, and costs ~28-41 ms/step.
        # Switched off here: same-seed rerun pairs of both SNP trainers
        # (Qwen3-8B) and the batched trainer (Qwen3-32B) stay
        # bit-identical without it, and fill-on/fill-off runs of the same
        # seed are themselves bit-identical
        # (results/snp_sweep/det_fill_off_check/summary.json).
        torch.utils.deterministic.fill_uninitialized_memory = False
        print("[run] torch.use_deterministic_algorithms(True) enabled "
              f"(CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}, "
              "fill_uninitialized_memory=False)")

    shared = _shared_kwargs(args)

    # Backend flag — only meaningful for TA / Suppress today (SNP already
    # trains on SDPA). Threaded through `shared` for the suppress branch
    # and added to the thought_anchors_compat dispatch below.
    shared["backend"] = args.backend

    if args.mode == "learn":
        # `learn.main` doesn't currently accept a backend kwarg; SNP +
        # attribution algorithms use SDPA already, so drop it for that path.
        shared.pop("backend", None)
        from expts.direct_answer_circuit_discovery.learn import main as learn_main
        learn_main(
            **shared,
            freeze_prompt_sentences=args.freeze_prompt_sentences,
            freeze_sentences_before=args.freeze_sentences_before,
            masking_algorithm=args.masking_algorithm,
            objective=args.objective,
            answer_bank_path=args.answer_bank_path,
            target_letter=args.target_letter,
            target_probs=(
                [float(x) for x in args.target_probs.split(",")]
                if isinstance(args.target_probs, str)
                else args.target_probs
            ),
            logit_margin_reduce=args.logit_margin_reduce,
            layers_to_analyse=_parse_layers(args.layers_to_analyse),
            mask_granularity=args.mask_granularity,
            pair_aggregation=args.pair_aggregation,
            ablate_non_target_layers=args.ablate_non_target_layers,
            num_ig_steps=args.num_ig_steps,
            no_negate_scores=args.no_negate_scores,
            include_zero_ablation=not args.no_include_zero_ablation,
            zero_ablation_epsilon=args.zero_ablation_epsilon,
            batch_chunk_size=args.batch_chunk_size,
            torch_compile=args.torch_compile,
            gradient_checkpointing=args.gradient_checkpointing,
            l0_lambda=args.l0_lambda,
            num_training_steps=args.num_training_steps,
            learning_rate=args.learning_rate,
            log_alpha_init=args.log_alpha_init,
            log_alpha_init_mask_path=args.log_alpha_init_mask_path,
            log_alpha_init_mask_alpha=args.log_alpha_init_mask_alpha,
            log_every=args.log_every,
            plot_every=args.plot_every,
            sparsity_loss_mode=args.sparsity_loss_mode,
            l0_normalize_hinge=args.l0_normalize_hinge,
            l0_warmup_frac=args.l0_warmup_frac,
            l0_ramp_frac=args.l0_ramp_frac,
            target_sparsity=args.target_sparsity,
            optimizer=args.optimizer,
            momentum=args.momentum,
            l0_lr_multiplier=args.l0_lr_multiplier,
            dropout_p=args.dropout_p,
            save_log_alpha=args.save_log_alpha,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            resume_from_checkpoint=args.resume_from_checkpoint,
            num_hc_samples_per_step=args.num_hc_samples_per_step,
            polyak_ema_log_alpha=args.polyak_ema_log_alpha,
            hc_beta_anneal=args.hc_beta_anneal,
            hc_beta_start=args.hc_beta_start,
            hc_beta_end=args.hc_beta_end,
            hc_beta_anneal_end_frac=args.hc_beta_anneal_end_frac,
            lr_schedule=args.lr_schedule,
            lr_min_ratio=args.lr_min_ratio,
            lr_plateau_patience=args.lr_plateau_patience,
            lr_plateau_factor=args.lr_plateau_factor,
            pid_kp=args.pid_kp,
            pid_ki=args.pid_ki,
            pid_kd=args.pid_kd,
            pid_mult_init=args.pid_mult_init,
            pid_ramp_end_frac=args.pid_ramp_end_frac,
            pid_max_target_sparsity=args.pid_max_target_sparsity,
            snapshot_sparsities=(
                [float(x) for x in args.snapshot_sparsities.split(",")]
                if isinstance(args.snapshot_sparsities, str)
                else args.snapshot_sparsities
            ),
            dcm_mask_init=args.dcm_mask_init,
            dcm_l0_optimizer=args.dcm_l0_optimizer,
            dcm_polarization=args.dcm_polarization,
            pid_snapshot_hold_steps=args.pid_snapshot_hold_steps,
            dcm_lr_init=args.dcm_lr_init,
            dcm_lr_warmup_frac=args.dcm_lr_warmup_frac,
        )
    else:
        from expts.direct_answer_circuit_discovery.suppress import main as supp_main
        supp_main(**shared, freeze_prompt_sentences=args.freeze_prompt_sentences)


if __name__ == "__main__":
    main()
