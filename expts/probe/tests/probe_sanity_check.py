import argparse
import json
import os
import pandas as pd
import torch
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.probe_utils import (
    LinearProbe,
    MLPProbe,
    ProbeCV,
    get_activations,
    AttentionProbe,
)
from utils.utils import clear_cuda, set_seed, MODEL_METADATA
from utils.prompt_utils import get_cot_prompt
from tqdm import tqdm


def make_prompt_mcq(base_data_dict: dict, tokenizer: AutoTokenizer):
    question = base_data_dict["question"]
    option_choices = base_data_dict["all_answers"]
    letter_choices = base_data_dict["all_letters"]
    formatted_question = f"{question}\n\nChoices:\n" + "\n".join(
        f"{letter}) {option}" for letter, option in zip(letter_choices, option_choices)
    )
    prompt_str = get_cot_prompt(tokenizer, formatted_question, multiple_choice=True)
    return prompt_str


def main(
    model_name: str = "gpt2",
    dataset_name: str = "AQuA",
    # probe experiment arguments
    probe_class: str = "linear",
    layer: int = 0,
    test_split: float = 0.1,
    cross_val_split: int = 5,
    epochs: int = 100,
    early_stopping: bool = False,
    patience: int = 10,
    learning_rate: float = 0.001,
    # experiment arguments
    seed: int = 42,
    # mlp probe-specific kwargs
    hidden_size: int = -1,
    num_layers: int = -1,
    min_entropy: float = 0.0,
):
    set_seed(seed)
    # 1. load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda", dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open("config.json") as f:
        config = json.load(f)
    streamlit_folder = config["save_locations"]["streamlit_folder"]
    probe_folder = config["save_locations"]["probe_folder"]
    model_nickname = MODEL_METADATA[model_name]["nickname"]
    example_ids = sorted(
        [
            filename.split(".")[0]
            for filename in os.listdir(
                f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}"
            )
            if filename != "base_data.json"
        ]
    )

    # Collect all data points as IID samples
    all_activations = []
    all_labels = []
    all_metadata = []  # For tracking predictions later

    for example_index in tqdm(example_ids, desc="Loading Activations"):
        question_id = int(example_index)
        with open(
            f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/base_data.json"
        ) as f:
            base_data = json.load(f)[question_id]
        outcome_df = pd.read_csv(
            f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/{example_index}.csv"
        )
        outcome_set = (
            outcome_df.groupby("outcome")["outcome_probability"]
            .sum()
            .sort_values(ascending=False)
            .index.values
        )
        timestamps = sorted(outcome_df.t.unique())

        # Get activations
        if dataset_name == "gpqa":
            prompt = make_prompt_mcq(base_data, tokenizer)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        full_token_ids = torch.tensor(
            prompt_token_ids + base_data["output_token_ids"], device=model.device
        ).unsqueeze(0)
        activations = get_activations(model, {"input_ids": full_token_ids}, layer=layer)

        # Compute entropy
        distribution = []
        for t in timestamps:
            distribution_at_t = []
            for outcome in outcome_set:
                row = outcome_df[
                    (outcome_df["outcome"] == outcome) & (outcome_df["t"] == t)
                ]
                if len(row) == 0:
                    distribution_at_t.append(0.0)
                elif len(row) == 1:
                    distribution_at_t.append(row.iloc[0]["outcome_probability"])
                else:
                    assert False, f"Only one row per outcome & t: {outcome} & {t}"
            distribution.append(distribution_at_t)

        d = Categorical(probs=torch.tensor(distribution))
        entropy = d.entropy()
        binary_labels = (entropy > min_entropy + 1e-6).float()

        # Add each timestamp as an IID data point
        for idx, t in enumerate(timestamps):
            all_activations.append(activations[t])
            all_labels.append(binary_labels[idx].item())
            all_metadata.append(
                {
                    "question_id": question_id,
                    "timestamp": t,
                    "entropy": entropy[idx].item(),
                }
            )

    del model
    clear_cuda()

    # Stack all data
    all_activations = torch.stack(all_activations)  # (N, hidden_dim)
    all_labels = torch.tensor(all_labels, dtype=all_activations.dtype)  # (N,)
    n_samples = len(all_labels)

    print(f"Total IID samples: {n_samples}")
    print(
        f"Label distribution: 0s={int((all_labels == 0).sum())}, 1s={int((all_labels == 1).sum())}"
    )

    # Random shuffle
    torch.manual_seed(seed)
    perm = torch.randperm(n_samples)
    all_activations = all_activations[perm]
    all_labels = all_labels[perm]
    all_metadata = [all_metadata[i] for i in perm.tolist()]

    # Train/test split
    test_size = int(n_samples * test_split) if test_split < 1 else int(test_split)

    activations_train = all_activations[test_size:]
    activations_test = all_activations[:test_size]
    labels_train = all_labels[test_size:]
    labels_test = all_labels[:test_size]
    metadata_test = all_metadata[:test_size]

    print(f"\nTrain size: {len(labels_train)}, Test size: {len(labels_test)}")
    print("Train inputs:", activations_train.dtype, activations_train.shape)
    print(
        f"Train label distribution: 0s={int((labels_train == 0).sum())}, 1s={int((labels_train == 1).sum())}"
    )
    print("Test inputs:", activations_test.dtype, activations_test.shape)
    print(
        f"Test label distribution: 0s={int((labels_test == 0).sum())}, 1s={int((labels_test == 1).sum())}"
    )

    # Setup probe
    if probe_class == "linear":
        probe_type = LinearProbe
        probe_kwargs = {}
    elif probe_class == "mlp":
        probe_type = MLPProbe
        probe_kwargs = {
            "hidden_size": hidden_size * activations_train.shape[-1],
            "num_layers": num_layers,
        }
    elif probe_class == "attention":
        probe_type = AttentionProbe
        probe_kwargs = {
            "d_proj": 512,
            "nhead": 1,
            "sliding_window": None,
            "max_length": 16384,
        }
    else:
        assert False, f"Probe type {probe_class} not implemented"

    # Train probe
    probe = ProbeCV(
        probe_type,
        n_split=cross_val_split,
        input_size=activations_train.shape[-1],
        output_size=1,
        epochs=epochs,
        device="cuda",
        early_stopping=early_stopping,
        patience=patience,
        loss_type="bce",
        learning_rate=learning_rate,
        **probe_kwargs,
    )
    probe.fit(activations_train, labels_train)

    # Evaluate
    train_loss = probe.score(activations_train, labels_train)
    test_loss = probe.score(activations_test, labels_test)

    train_pred_probs = torch.sigmoid(probe.pred(activations_train).squeeze(-1))
    test_pred_probs = torch.sigmoid(probe.pred(activations_test).squeeze(-1))
    train_pred_labels = (train_pred_probs > 0.5).float()
    test_pred_labels = (test_pred_probs > 0.5).float()

    train_accuracy = (
        (train_pred_labels == labels_train.to(train_pred_labels.device))
        .float()
        .mean()
        .item()
    )
    test_accuracy = (
        (test_pred_labels == labels_test.to(test_pred_labels.device))
        .float()
        .mean()
        .item()
    )

    print(f"\nTrain Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
    print(f"Train Acc={train_accuracy:.4f}, Test Acc={test_accuracy:.4f}")

    # Save results
    # results = {
    #     "hyperparameters": {
    #         "probe_class": probe_class,
    #         "layer": layer,
    #         "epochs": epochs,
    #         "early_stopping": early_stopping,
    #         "patience": patience,
    #         "learning_rate": learning_rate,
    #         "seed": seed,
    #         "num_layers": num_layers,
    #         "loss_type": "bce",
    #         "split_by": "iid_random",
    #         "test_split": test_split,
    #     },
    #     "metrics": {
    #         "train_loss": train_loss,
    #         "test_loss": test_loss,
    #         "train_accuracy": train_accuracy,
    #         "test_accuracy": test_accuracy,
    #         "n_train": len(labels_train),
    #         "n_test": len(labels_test),
    #     },
    #     "predictions": [],
    # }

    # pred_probs = test_pred_probs.cpu()
    # pred_labels = test_pred_labels.cpu()
    # for i, meta in enumerate(metadata_test):
    #     results["predictions"].append({
    #         "question_id": meta["question_id"],
    #         "timestamp": meta["timestamp"],
    #         "true_entropy": meta["entropy"],
    #         "true_label": int(labels_test[i].item()),
    #         "pred_prob": float(pred_probs[i]),
    #         "pred_label": int(pred_labels[i]),
    #     })

    # os.makedirs(f"{probe_folder}/{model_nickname}/{dataset_name.lower()}", exist_ok=True)
    # output_filename = f"results-classifier-iid-{probe_class}-layer{layer}-epochs{epochs}.json"
    # with open(
    #     f"{probe_folder}/{model_nickname}/{dataset_name.lower()}/{output_filename}",
    #     "w+",
    # ) as f:
    #     json.dump(results, f, indent=2)
    # print(f"\nSaved to {probe_folder}/{model_nickname}/{dataset_name.lower()}/{output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="AQuA")
    parser.add_argument("--probe_class", type=str, default="linear")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--cross_val_split", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=-1)
    parser.add_argument("--num_layers", type=int, default=-1)
    parser.add_argument("--min_entropy", type=float, default=0.0)
    args = parser.parse_args()
    main(**vars(args))
