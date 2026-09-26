"""Train a Reasoning Theater probe on cached per-layer activations.

The probe is trained with random end-truncation (each rollout becomes "predict the final
answer from a random prefix"). Train/test split is by question (prompt_id), not rollout,
to prevent prefix leakage when multiple rollouts share a prompt.

Run:
    uv run python -m expts.reasoning_theater_probes.train_probe \
        --config expts/reasoning_theater_probes/configs/qwen3_8b_gpqa_diamond.yaml
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.expt_config import load_config
from utils.utils import MODEL_METADATA, set_seed

from expts.reasoning_theater_probes.dataset import (
    ProbeDataset,
    collate,
    load_meta,
    num_classes_for,
    split_by_question,
)
from expts.reasoning_theater_probes.probes import build_probe


@torch.no_grad()
def eval_accuracy(probe: torch.nn.Module, loader: DataLoader, device: str) -> float:
    probe.eval()
    correct = 0
    total = 0
    for activations, labels, lengths, _ in loader:
        activations = activations.to(device)
        labels = labels.to(device)
        logits = probe(activations, lengths=lengths)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.numel()
    probe.train()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = cfg.get("device", "cuda")

    label_type: str = cfg["label_type"]
    probe_class: str = cfg.get("probe_class", "linear")
    epochs: int = cfg.get("epochs", 50)
    batch_size: int = cfg.get("batch_size", 4)
    lr: float = cfg.get("learning_rate", 1e-3)
    weight_decay: float = cfg.get("weight_decay", 0.0)
    test_frac: float = cfg.get("test_frac", 0.2)

    model_name = cfg["model_name"]
    layer = cfg["layer"]
    data_path = cfg["data_path"]
    activation_root = Path(cfg["activation_dir"])
    output_root = Path(cfg["output_dir"])

    nickname = MODEL_METADATA[model_name]["nickname"]
    dataset_nick = Path(data_path).stem
    activation_dir = activation_root / nickname / dataset_nick / f"layer{layer}"
    out_dir = output_root / nickname / dataset_nick / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{probe_class}_{label_type}"

    entries = load_meta(activation_dir)
    train_entries, test_entries = split_by_question(entries, test_frac, seed)
    print(f"layer={layer}  train={len(train_entries)} rollouts  test={len(test_entries)} rollouts")

    train_ds = ProbeDataset(train_entries, activation_dir, label_type, training=True)
    test_ds = ProbeDataset(test_entries, activation_dir, label_type, training=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    sample_act, _, _ = train_ds[0]
    in_features = sample_act.shape[-1]
    out_features = num_classes_for(label_type)
    probe = build_probe(
        probe_class,
        in_features,
        out_features,
        mlp=cfg.get("mlp", False),
        mlp_hidden_dim=cfg.get("mlp_hidden_dim", 32),
    ).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state: dict | None = None
    history = []
    for epoch in range(epochs):
        probe.train()
        ep_loss, n = 0.0, 0
        for activations, labels, lengths, _ in tqdm(train_loader, desc=f"epoch {epoch}"):
            activations = activations.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = probe(activations, lengths=lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n += 1
        train_acc = eval_accuracy(probe, train_loader, device)
        test_acc = eval_accuracy(probe, test_loader, device)
        history.append(
            {"epoch": epoch, "train_loss": ep_loss / max(n, 1), "train_acc": train_acc, "test_acc": test_acc}
        )
        print(
            f"epoch {epoch}: loss={ep_loss / max(n, 1):.4f}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in probe.state_dict().items()})

    torch.save(best_state, out_dir / f"probe_{suffix}.pt")
    with open(out_dir / f"history_{suffix}.json", "w") as f:
        json.dump(
            {
                "config": cfg,
                "history": history,
                "best_test_acc": best_acc,
                "test_prompt_ids": sorted({e["prompt_id"] for e in test_entries}),
            },
            f,
            indent=2,
        )
    print(f"best test acc = {best_acc:.3f} | saved probe to {out_dir / f'probe_{suffix}.pt'}")


if __name__ == "__main__":
    main()
