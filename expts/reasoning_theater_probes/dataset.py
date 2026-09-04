"""Dataset and label utilities for Reasoning Theater probing.

Activations are cached per rollout as `(T, D)` tensors covering both the prompt and the
generated CoT. During training we apply paper-faithful random end-truncation, restricted
to positions inside the generated portion so the probe always sees at least the full
prompt before being asked to predict the final answer.
"""

import json
import random
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def encode_label(entry: dict, label_type: str) -> int:
    if label_type == "model_answer":
        return LETTER_TO_IDX[entry["clean_answer"]]
    if label_type == "correct_answer":
        return LETTER_TO_IDX[entry["correct_letter"]]
    if label_type == "model_correct":
        return int(entry["clean_answer"] == entry["correct_letter"])
    raise ValueError(f"Unknown label_type: {label_type}")


def num_classes_for(label_type: str) -> int:
    return 2 if label_type == "model_correct" else 4


class ProbeDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[dict],
        activation_dir: str | Path,
        label_type: str,
        training: bool = False,
    ) -> None:
        self.entries = list(entries)
        self.activation_dir = Path(activation_dir)
        self.label_type = label_type
        self.training = training

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        activation = torch.load(self.activation_dir / entry["filename"], map_location="cpu")
        if activation.dtype != torch.float32:
            activation = activation.float()

        gen_start = max(0, entry["prompt_len"] - 1)
        activation = activation[gen_start:]

        if self.training and activation.shape[0] > 1:
            end_idx = random.randint(1, activation.shape[0])
            activation = activation[:end_idx]

        label = encode_label(entry, self.label_type)
        meta = {
            "prompt_id": entry["prompt_id"],
            "rollout_id": entry["rollout_id"],
            "gen_start": gen_start,
        }
        return activation, label, meta


def collate(batch):
    activations, labels, metas = zip(*batch)
    max_len = max(act.shape[0] for act in activations)
    hidden_dim = activations[0].shape[1]
    dtype = activations[0].dtype
    padded = torch.zeros(len(activations), max_len, hidden_dim, dtype=dtype)
    lengths = []
    for i, act in enumerate(activations):
        seq_len = act.shape[0]
        padded[i, :seq_len] = act
        lengths.append(seq_len)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, label_tensor, lengths, list(metas)


def load_meta(activation_dir: str | Path) -> list[dict]:
    activation_dir = Path(activation_dir)
    with open(activation_dir / "meta.json") as f:
        return json.load(f)


def split_by_question(entries: list[dict], test_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out `test_frac` of unique prompt_ids for the test split.

    Splitting by prompt_id (not by rollout) is critical: rollouts from the same question
    share most of their prefix activations, so a rollout-level split would leak.
    """
    rng = random.Random(seed)
    prompt_ids = sorted({e["prompt_id"] for e in entries})
    rng.shuffle(prompt_ids)
    n_test = max(1, int(len(prompt_ids) * test_frac))
    test_ids = set(prompt_ids[:n_test])
    train = [e for e in entries if e["prompt_id"] not in test_ids]
    test = [e for e in entries if e["prompt_id"] in test_ids]
    return train, test
