"""Learn a circuit mask and evaluate it — drop-in replacement for the
original expts/learn_circuit.py.

Calls learn_circuit.main() then evaluate_mask.evaluate() sequentially.
"""

import argparse

from expts.circuit_discovery.learn_circuit import main as learn_circuit_main
from expts.circuit_discovery.learn_circuit import _parse_layers_arg
from expts.circuit_discovery.evaluate_mask import evaluate, DEFAULT_SPARSITIES
from utils.completion_cache import DEFAULT_CACHE_DIR
from utils.circuit_discovery.factory import get_available_algorithms


def main(**kwargs):
    # Pop eval-only args before forwarding to learn_circuit
    sparsities = kwargs.pop("sparsities", None) or list(DEFAULT_SPARSITIES)
    num_random_samples = kwargs.pop("num_random_samples", 5)
    device = kwargs.get("device", "cuda")
    cache_dir = kwargs.get("cache_dir", DEFAULT_CACHE_DIR)

    mask_path = learn_circuit_main(**kwargs)

    if mask_path is None:
        # Discovery was aborted (e.g. user declined cost warning)
        return

    evaluate(
        mask_path=mask_path,
        sparsities=sparsities,
        num_random_samples=num_random_samples,
        device=device,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learn a circuit mask and evaluate at multiple sparsity thresholds"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file. CLI args override config values.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model for vLLM generation (branches).",
    )
    parser.add_argument(
        "--model_to_analyse",
        type=str,
        default=None,
        help="Model loaded with eager attention for circuit discovery. "
        "Defaults to --model_name if not specified.",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_new_branches", type=int, default=8)
    parser.add_argument(
        "--masking_algorithm",
        choices=get_available_algorithms(),
        default="nodewise_attribution",
    )
    parser.add_argument(
        "--pair_aggregation",
        choices=["sum", "mean", "median", "max"],
        default="mean",
        help="How to aggregate token-pair AP+IG scores into sentence-pair mask scores.",
    )
    parser.add_argument(
        "--mask_granularity",
        choices=["head", "layer", "pair"],
        default="head",
        help="Score granularity: 'head' (per-head), 'layer' (shared across heads), "
        "'pair' (shared across layers and heads).",
    )
    parser.add_argument(
        "--analysis_timestep",
        type=int,
        default=None,
        help="Token index for analysis (default: prompt length)",
    )
    parser.add_argument(
        "--analysis_sentence_step",
        type=int,
        default=None,
        help="Sentence index (counted from start of prompt+base) at whose end "
        "the analysis boundary is placed. Requires --data_path + --prompt_index. "
        "If both this and --analysis_timestep are set, the sentence-based one "
        "wins (with a WARNING). Sample-and-cut from a fresh base is not yet "
        "supported.",
    )
    parser.add_argument(
        "--base_answer_type",
        choices=["stored", "correct", "incorrect", "mode"],
        default="stored",
        help="Which path within the data-collection record to use as the base. "
        "'stored' uses record['output_token_ids'] directly. Others may "
        "retokenize an alternate text (lossy).",
    )
    parser.add_argument(
        "--objective",
        choices=["kl_divergence", "log_prob", "answer_kl", "reward_gap"],
        default="kl_divergence",
        help="Local: kl_divergence, log_prob (per-token). "
        "Global: answer_kl (Obj 1, faithfulness), reward_gap (Obj 2, reward).",
    )
    parser.add_argument(
        "--layers_to_analyse",
        type=_parse_layers_arg,
        nargs="+",
        default=[8, 12, 16, 20, 24],
        help="Layer indices to analyze, or 'all' for every layer.",
    )
    parser.add_argument("--sentence_gap", type=int, default=1)
    parser.add_argument("--sentence_chunk", type=int, default=1)
    parser.add_argument(
        "--mask_mode",
        choices=["prefix", "generation", "both"],
        default="prefix",
        help="Which query-key region to learn: prefix (MASK 1), "
        "generation (MASK 2), or both.",
    )
    parser.add_argument(
        "--ablate_non_target_layers",
        action="store_true",
        help="Ablate all attention heads in layers outside --layers_to_analyse",
    )
    parser.add_argument(
        "--no_renormalize_masked_attention",
        dest="renormalize_masked_attention",
        action="store_false",
        help="Do not renormalize post-softmax attention after applying the mask.",
    )
    parser.add_argument("--num_ig_steps", type=int, default=10)
    parser.add_argument(
        "--num_random_samples",
        type=int,
        default=5,
        help="Number of random score masks (K) to sample for baseline comparison.",
    )
    parser.add_argument(
        "--no_negate_scores",
        action="store_true",
        help="Store raw IG scores (positive = increases KL). "
        "Default negates so positive = helps retention.",
    )
    parser.add_argument("--max_sampling_tokens", type=int, default=150,
        help="Max tokens for vLLM generation (base completion and branches).")
    parser.add_argument("--num_tokens_to_analyse", type=int, default=None,
        help="Number of continuation tokens to use for local KL objectives. "
        "Defaults to max_sampling_tokens. Set lower to analyse only the "
        "beginning of each branch while still sampling to completion.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/circuit_discovery")
    parser.add_argument(
        "--min_sentence_length",
        type=int,
        default=10,
        help="Minimum number of tokens for a sentence (after splitting).",
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=DEFAULT_SPARSITIES,
        help="Target sparsity levels (0-1) for evaluation. Thresholds are "
        "computed dynamically from the learned mask scores.",
    )
    parser.add_argument(
        "--reward_type",
        choices=["none", "correctness", "cot_length"],
        default="none",
        help="Reward type for reward-weighted circuit discovery.",
    )
    parser.add_argument(
        "--correct_answer",
        type=str,
        default=None,
        help="Ground truth answer string (for correctness reward).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to collection JSON with question/answer records.",
    )
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=None,
        help="Index into collection JSON to load question + correct_answer.",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["open ended", "multiple choice", "alignment"],
        default="open ended",
        help="Dataset type for answer parsing.",
    )
    parser.add_argument(
        "--answer_only",
        action="store_true",
        help="Restrict position mask to answer tokens only (\\boxed{...}).",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="meta-llama/llama-3.2-3b-instruct",
        help="Model for LLM-based answer judging (OpenRouter).",
    )
    parser.add_argument(
        "--judge_answers",
        action="store_true",
        help="Use LLM judge to cluster branch answers by mathematical "
        "equivalence (for global objectives). Requires OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default=None,
        help="Custom output file name (saved under --output_dir). "
        "Auto-appends .json if missing. Overrides the default naming convention.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory for caching vLLM completions.",
    )
    parser.add_argument(
        "--batch_chunk_size",
        type=int,
        default=None,
        help="Number of continuations per forward-pass chunk in flash patching. "
        "Lower values reduce GPU memory usage (default: algorithm's own default, typically 4).",
    )
    parser.add_argument(
        "--importance_sampling_method",
        choices=["snis", "geometric_mean", "tempered_snis"],
        default="snis",
        help="Importance-sampling reweighting method. "
        "'snis' is the standard self-normalised estimator; "
        "'geometric_mean' divides each chain's log-ratio by its length "
        "before softmax, mitigating SNIS collapse on long chains; "
        "'tempered_snis' divides each chain's log-ratio by a fixed scalar "
        "temperature (set via --importance_sampling_temperature). "
        "See notes/reward_gap_goodhart.md, notes/geometric_mean_collapse.md.",
    )
    parser.add_argument(
        "--importance_sampling_temperature",
        type=float,
        default=None,
        help="Scalar temperature T for --importance_sampling_method tempered_snis. "
        "T=1 recovers SNIS, T->inf recovers uniform. "
        "See notes/geometric_mean_collapse.md.",
    )
    parser.add_argument(
        "--l0_lambda",
        type=float,
        default=None,
        help="L0 sparsity penalty weight for nodewise_subnetwork_probing_sdpa. "
        "Ignored by other algorithms.",
    )
    parser.add_argument(
        "--num_training_steps", type=int, default=None,
        help="Adam steps for nodewise_subnetwork_probing_sdpa.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Adam learning rate for nodewise_subnetwork_probing_sdpa.",
    )
    parser.add_argument(
        "--log_alpha_init", type=float, default=None,
        help="Initial Hard-Concrete location parameter for SNP.",
    )
    parser.add_argument(
        "--log_every", type=int, default=None,
        help="SNP: record training-curve point every N steps.",
    )
    parser.add_argument(
        "--plot_every", type=int, default=None,
        help="SNP: overwrite training-curve PDFs every N steps.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable activation checkpointing on the analysis model to reduce "
        "activation memory during IG backward. Forces use_cache=False.",
    )
    # First parse to check for --config
    args, _ = parser.parse_known_args()
    if args.config:
        from utils.expt_config import load_config

        config = load_config(args.config)
        # Apply config values as new argparse defaults; CLI args will override
        parser.set_defaults(**{k: v for k, v in config.items() if k != "config"})
    # Re-parse with config-informed defaults
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("config", None)
    main(**kwargs)
