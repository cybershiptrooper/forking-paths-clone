"""Cache per-layer hidden states for the base rollouts in a data collection.

Output layout (under `activation_dir`):
    <activation_dir>/<model_nickname>/<dataset_nickname>/layer<L>/
        meta.json                         # list of per-rollout entries
        <prompt_id>_<rollout_id>.pt       # (T, D) tensor in `storage_dtype`

The cached tensor includes the full sequence (prompt + generated CoT). `prompt_len` in
the meta entry tells consumers where the generated region starts.

Run:
    uv run python -m expts.reasoning_theater_probes.cache_activations \
        --config expts/reasoning_theater_probes/configs/qwen3_8b_gpqa_diamond.yaml
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from utils.expt_config import load_config
from utils.utils import MODEL_METADATA, clear_cuda, set_seed


@torch.no_grad()
def extract_layer(model: torch.nn.Module, layer: int, input_ids: torch.LongTensor) -> torch.Tensor:
    """Return hidden states from `layer` for a single sequence — shape (T, D)."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        model(
            input_ids=input_ids.to(model.device).unsqueeze(0),
            output_hidden_states=False,
            use_cache=False,
        )
    finally:
        handle.remove()
    return captured["h"].squeeze(0).to("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.get("seed", 42))
    model_name = cfg["model_name"]
    layer = cfg["layer"]
    data_path = cfg["data_path"]
    activation_root = Path(cfg["activation_dir"])
    storage_dtype = getattr(torch, cfg.get("storage_dtype", "float16"))
    max_examples = cfg.get("max_examples")

    nickname = MODEL_METADATA[model_name]["nickname"]
    dataset_nick = Path(data_path).stem
    out = activation_root / nickname / dataset_nick / f"layer{layer}"
    out.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        collection = json.load(f)
    if max_examples:
        collection = collection[: int(max_examples)]

    print(f"Loading {model_name} (layer {layer}, {len(collection)} rollouts)...")
    # sdpa (not eager) so the 20k+-token GPQA rollouts fit in memory; eager's full
    # attention matrix is ~30 GB at seq_len=23k for Qwen3-8B's 32 heads. The layer-output
    # hook used here doesn't depend on attention kernel internals.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()

    meta = []
    for entry in tqdm(collection, desc="caching"):
        token_ids = torch.tensor(
            entry["prompt_token_ids"] + entry["output_token_ids"], dtype=torch.long
        )
        h = extract_layer(model, layer, token_ids).to(storage_dtype)
        filename = f"{entry['prompt_id']}_{entry['rollout_id']}.pt"
        torch.save(h, out / filename)
        meta.append(
            {
                "filename": filename,
                "prompt_id": int(entry["prompt_id"]),
                "rollout_id": int(entry["rollout_id"]),
                "is_base": True,
                "clean_answer": entry["clean_answer"],
                "correct_letter": entry["correct_letter"],
                "all_letters": entry["all_letters"],
                "prompt_len": len(entry["prompt_token_ids"]),
                "total_len": int(h.shape[0]),
            }
        )

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    del model
    clear_cuda()
    print(f"Saved {len(meta)} activations to {out}")


if __name__ == "__main__":
    main()
