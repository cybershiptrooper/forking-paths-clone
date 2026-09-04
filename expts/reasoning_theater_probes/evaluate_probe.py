"""Apply a trained probe at every (or sampled) token position in the generated CoT.

Produces a per-token decodability record so you can plot when the model's final answer
becomes linearly readable from intermediate hidden states. Output is a JSON list of rows
(one per (rollout, position)) — readable by pandas / streamlit.

Run:
    uv run python -m expts.reasoning_theater_probes.evaluate_probe \
        --config expts/reasoning_theater_probes/configs/qwen3_8b_gpqa_diamond.yaml \
        --num_positions 200
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from utils.expt_config import load_config
from utils.utils import MODEL_METADATA

from expts.reasoning_theater_probes.dataset import encode_label, load_meta, num_classes_for
from expts.reasoning_theater_probes.probes import AttentionProbe, LinearProbe, build_probe


@torch.no_grad()
def attention_logits_at_all_prefixes(probe: AttentionProbe, activation: torch.Tensor) -> torch.Tensor:
    """AttentionProbe logits at every prefix length, computed in O(T) via cumulative softmax.

    Returns a tensor of shape `(T, num_classes)` where row `L` is the probe's output when
    applied to `activation[:L + 1]`. Equivalent to looping `probe(activation[:L+1])` for
    every L, but folds the softmax-weighted sum into two `cumsum`s.
    """
    attn_logits = probe.q(activation).squeeze(-1)
    exp_logits = torch.exp(attn_logits - attn_logits.max())
    Z = torch.cumsum(exp_logits, dim=0)
    v = probe.v_up(activation) if probe.mlp else probe.v(activation)
    A = torch.cumsum(exp_logits.unsqueeze(-1) * v, dim=0)
    out = A / Z.unsqueeze(-1)
    if probe.mlp:
        out = probe.v_down(probe.v_relu(out))
    return out


@torch.no_grad()
def predict_at_positions(probe: torch.nn.Module, activation: torch.Tensor, positions: list[int]) -> torch.Tensor:
    """Logits with shape (len(positions), num_classes) — one row per prefix length."""
    if isinstance(probe, LinearProbe):
        return probe.linear(activation[positions])
    if isinstance(probe, AttentionProbe):
        return attention_logits_at_all_prefixes(probe, activation)[positions]
    raise ValueError(f"Unsupported probe type: {type(probe).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--num_positions",
        type=int,
        default=200,
        help="Evenly-spaced positions per rollout (0 = every generated token).",
    )
    parser.add_argument(
        "--prompt_ids",
        type=str,
        default=None,
        help="Override which prompt_ids to evaluate (default: held-out test set from training history).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = cfg.get("device", "cuda")
    label_type = cfg["label_type"]
    probe_class = cfg.get("probe_class", "linear")
    model_name = cfg["model_name"]
    layer = cfg["layer"]
    data_path = cfg["data_path"]
    activation_root = Path(cfg["activation_dir"])
    output_root = Path(cfg["output_dir"])

    nickname = MODEL_METADATA[model_name]["nickname"]
    dataset_nick = Path(data_path).stem
    activation_dir = activation_root / nickname / dataset_nick / f"layer{layer}"
    out_dir = output_root / nickname / dataset_nick / f"layer{layer}"
    suffix = f"{probe_class}_{label_type}"

    history_path = out_dir / f"history_{suffix}.json"
    history = json.loads(history_path.read_text())
    if args.prompt_ids:
        target_pids = {int(p) for p in args.prompt_ids.split(",")}
    else:
        target_pids = set(history["test_prompt_ids"])

    entries = [e for e in load_meta(activation_dir) if e["prompt_id"] in target_pids]
    print(f"evaluating {len(entries)} rollouts at layer {layer}")

    sample_act = torch.load(activation_dir / entries[0]["filename"], map_location="cpu").float()
    in_features = sample_act.shape[-1]
    out_features = num_classes_for(label_type)
    probe = build_probe(
        probe_class,
        in_features,
        out_features,
        mlp=cfg.get("mlp", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 32),
    )
    probe.load_state_dict(torch.load(out_dir / f"probe_{suffix}.pt", map_location="cpu"))
    probe.to(device).eval()

    rows: list[dict] = []
    for entry in tqdm(entries, desc="evaluating"):
        activation = torch.load(activation_dir / entry["filename"], map_location=device).float()
        T = activation.shape[0]
        gen_start = max(0, entry["prompt_len"] - 1)
        n_gen = T - gen_start
        if args.num_positions == 0 or n_gen <= args.num_positions:
            positions = list(range(gen_start, T))
        else:
            step = max(1, n_gen // args.num_positions)
            positions = list(range(gen_start, T, step))[: args.num_positions]

        logits = predict_at_positions(probe, activation, positions)
        probs = torch.softmax(logits, dim=-1).cpu()
        preds = probs.argmax(dim=-1)
        true_label = encode_label(entry, label_type)

        for pos, prob_row, pred in zip(positions, probs, preds):
            rows.append(
                {
                    "prompt_id": entry["prompt_id"],
                    "rollout_id": entry["rollout_id"],
                    "t": int(pos),
                    "frac_t": (pos - gen_start) / max(n_gen - 1, 1),
                    "true_label": int(true_label),
                    "pred_label": int(pred),
                    "prob_true_label": float(prob_row[true_label]),
                }
            )

    out_path = out_dir / f"token_eval_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f)
    print(f"saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
