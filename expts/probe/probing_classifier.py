import argparse
import datetime
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
    single_test_set: bool = False,
):
    set_seed(seed)
    # 1. load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda", dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 2. load all forking paths data & collect activations
    # - per question: t -> entropy
    # - activation @ res. stream, t, layer -> entropy @ t
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
            if filename != "base_data.json"  # ignore base data
        ]
    )
    # ids_to_use = ["11", "13", "16", "17", "18", "20", "21", "23", "26", "27"]
    # example_ids = ids_to_use

    probe_data = {
        "t": [],  # (# questions, T)
        "activation": [],  # (# questions, T, hidden dim)
        "entropy": [],  # (# questions, T)
        "label": [],  # (# questions, T) - binary labels: 0 if entropy=0, 1 if entropy>0
    }
    for example_index in tqdm(example_ids, desc="Loading Activations"):
        # load base data
        with open(
            f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/base_data.json"
        ) as f:
            base_data = json.load(f)[int(example_index)]
        # load distribution
        outcome_df = pd.read_csv(
            f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/{example_index}.csv"
        )
        outcome_set = (
            outcome_df.groupby("outcome")["outcome_probability"]
            .sum()
            .sort_values(ascending=False)
            .index.values
        )  # (O,)
        timestamps = sorted(outcome_df.t.unique())  # (T,)
        probe_data["t"].append(timestamps)

        # a. get activations
        if dataset_name == "gpqa":
            prompt = make_prompt_mcq(base_data, tokenizer)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        full_token_ids = torch.tensor(
            prompt_token_ids + base_data["output_token_ids"], device=model.device
        ).unsqueeze(0)
        activations = get_activations(
            model, {"input_ids": full_token_ids}, layer=layer
        )  # (tokens, hidden dim)
        activations_per_t = activations[timestamps]  # (T, hidden dim)
        probe_data["activation"].append(activations_per_t)

        # b. compute entropy
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
                    assert (
                        False
                    ), f"Only one row per outcome & t: {outcome} & {t}; #{len(row)}"
            distribution.append(distribution_at_t)

        d = Categorical(probs=torch.tensor(distribution))
        entropy = d.entropy()
        probe_data["entropy"].append(entropy)

        # c. compute binary labels: 0 if entropy=0, 1 if entropy>0
        binary_labels = (entropy > min_entropy + 1e-6).float()
        probe_data["label"].append(binary_labels)

    del model  # done with model once activations are collected!
    clear_cuda()

    # 2. create train-test split
    # - hold out questions from dataset
    results = {
        "hyperparameters": {
            "probe_class": probe_class,
            "layer": layer,
            "epochs": epochs,
            "early_stopping": early_stopping,
            "patience": patience,
            "learning_rate": learning_rate,
            "seed": seed,
            "num_layers": num_layers,
            "loss_type": "bce",
            "single_test_set": single_test_set,
        },
        "metrics": {
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
        },
        "predictions": [],
    }

    split_size = (
        int(len(probe_data["label"]) * test_split) if test_split < 1 else int(test_split)
    )

    num_questions = len(probe_data["label"])

    if single_test_set:
        # Randomly sample test indices once
        torch.manual_seed(seed)
        all_indices = torch.randperm(num_questions).tolist()
        test_indices = set(all_indices[:split_size])
        split_ranges = [(test_indices, None)]  # Single split with random test indices
    else:
        # K-fold: rotate through all questions as test sets
        split_ranges = [
            (
                set(range(split_index, min(split_index + split_size, num_questions))),
                split_index,
            )
            for split_index in range(0, num_questions, split_size)
        ]

    for test_indices, split_index in split_ranges:
        activations, labels, entropy = (
            probe_data["activation"],
            probe_data["label"],
            probe_data["entropy"],
        )

        # Split based on test indices
        activations_test = [
            activations[i] for i in range(num_questions) if i in test_indices
        ]
        labels_test = [labels[i] for i in range(num_questions) if i in test_indices]
        entropy_test = [entropy[i] for i in range(num_questions) if i in test_indices]
        activations_train = [
            activations[i] for i in range(num_questions) if i not in test_indices
        ]
        labels_train = [labels[i] for i in range(num_questions) if i not in test_indices]
        entropy_train = [
            entropy[i] for i in range(num_questions) if i not in test_indices
        ]

        # Keep track of which question IDs are in the test set (in order)
        test_question_ids = [i for i in range(num_questions) if i in test_indices]

        activations_train = torch.cat(
            activations_train
        )  # (T * num_questions, hidden dim)
        activations_test = torch.cat(activations_test)  # (T * num_questions, hidden dim)
        labels_train = torch.cat(labels_train).to(
            activations_train.dtype
        )  # (T * num_questions,)
        labels_test = torch.cat(labels_test).to(
            activations_test.dtype
        )  # (T * num_questions,)
        print("Train inputs:", activations_train.dtype, activations_train.shape)
        print("Train labels:", labels_train.dtype, labels_train.shape)
        print(
            "Train label distribution: 0s:",
            (labels_train == 0).sum().item(),
            "1s:",
            (labels_train == 1).sum().item(),
        )
        print("Test inputs:", activations_test.dtype, activations_test.shape)
        print("Test labels:", labels_test.dtype, labels_test.shape)
        print(
            "Test label distribution: 0s:",
            (labels_test == 0).sum().item(),
            "1s:",
            (labels_test == 1).sum().item(),
        )

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
                "nhead": 8,
                "sliding_window": None,
                "max_length": 16384,
            }
        else:
            assert False, f"Probe type {probe_class} not implemented"

        # 3. train probe with BCE loss (binary classification)
        probe = ProbeCV(
            probe_type,
            n_split=cross_val_split,
            input_size=activations_train.shape[-1],
            output_size=1,
            epochs=epochs,
            device="cuda",
            early_stopping=early_stopping,
            patience=patience,
            loss_type="bce",  # Binary cross-entropy for classification
            learning_rate=learning_rate,
            **probe_kwargs,
        )
        probe.fit(activations_train, labels_train)

        # 4. evaluate probe on test questions
        train_loss = probe.score(activations_train, labels_train)
        test_loss = probe.score(activations_test, labels_test)

        # Compute accuracy
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

        results["metrics"]["train_loss"].append(train_loss)
        results["metrics"]["test_loss"].append(test_loss)
        results["metrics"]["train_accuracy"].append(train_accuracy)
        results["metrics"]["test_accuracy"].append(test_accuracy)

        split_name = "random" if single_test_set else f"{split_index}"
        print(
            f"Split {split_name}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}"
        )
        print(
            f"Split {split_name}: Train Acc={train_accuracy:.4f}, Test Acc={test_accuracy:.4f}"
        )

        # 5. save probe results
        pred_probs = test_pred_probs.cpu()  # (T * num_questions,)
        pred_labels = test_pred_labels.cpu()
        t_index = 0
        for question_id in test_question_ids:
            ts = probe_data["t"][question_id]
            true_entropy = probe_data["entropy"][question_id]
            true_labels = probe_data["label"][question_id]
            pred_probs_for_q = pred_probs[t_index : t_index + len(ts)]
            pred_labels_for_q = pred_labels[t_index : t_index + len(ts)]
            t_index += len(ts)  # offset by # of timestamps in each question
            assert len(true_labels) == len(ts) and len(true_labels) == len(
                pred_labels_for_q
            ), f"Ts, true labels and pred labels must be same length: {len(ts)}, {len(true_labels)}, {len(pred_labels_for_q)}"
            results["predictions"].append(
                {
                    "question_id": int(question_id),
                    "t": [int(t) for t in ts],
                    "true_entropy": [float(e) for e in true_entropy],
                    "true_label": [int(l) for l in true_labels],
                    "pred_prob": [float(p) for p in pred_probs_for_q],
                    "pred_label": [int(l) for l in pred_labels_for_q],
                }
            )

    os.makedirs(f"{probe_folder}/{model_nickname}/{dataset_name.lower()}", exist_ok=True)
    with open(
        f"{probe_folder}/{model_nickname}/{dataset_name.lower()}/results-classifier-{probe_class}-layer{layer}-epochs{epochs}.json",
        "w+",
    ) as f:
        json.dump(results, f, indent=2)


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
    parser.add_argument("--single_test_set", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
