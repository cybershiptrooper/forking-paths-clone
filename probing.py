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
    AttentionProbe,
    ProbeCV,
    get_activations,
)
from utils.utils import clear_cuda, set_seed, MODEL_METADATA
from utils.prompt_utils import get_cot_prompt


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
    model_name : str = "gpt2",
    dataset_name : str = "AQuA",
    # probe experiment arguments
    probe_class : str = "linear",
    layer : int = 0,
    test_split : float = 0.1,
    cross_val_split : int = 5,
    epochs : int = 100,
    early_stopping : bool = False,
    patience : int = 10,
    learning_rate : float = 0.001,
    # experiment arguments
    seed : int = 42,
    # mlp probe-specific kwargs
    hidden_size : int = -1,
    num_layers : int = -1 
):
    set_seed(seed)
    # 1. load model
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 2. load all forking paths data & collect activations
    # - per question: t -> entropy
    # - activation @ res. stream, t, layer -> entropy @ t
    with open('config.json') as f:
        config = json.load(f)
    streamlit_folder = config['save_locations']['streamlit_folder']
    probe_folder = config['save_locations']['probe_folder']
    model_nickname = MODEL_METADATA[model_name]['nickname']
    example_ids = sorted([
        filename.split('.')[0] for filename in os.listdir(f'{streamlit_folder}/{model_nickname}/{dataset_name.lower()}')
        if filename != "base_data.json" # ignore base data
    ])
    # ids_to_use = ["11", "13", "16", "17", "18", "20", "21", "23", "26", "27"]
    # example_ids = ids_to_use

    probe_data = {
        't': [], # (# questions, T)
        'activation': [], # (# questions, T, hidden dim)
        'entropy': [] # (# questions, T)
    }
    for example_index in example_ids:
        # load base data
        with open(f'{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/base_data.json') as f:
            base_data = json.load(f)[int(example_index)]
        # load distribution
        outcome_df = pd.read_csv(f"{streamlit_folder}/{model_nickname}/{dataset_name.lower()}/{example_index}.csv")
        outcome_set = outcome_df.groupby('outcome')['outcome_probability'].sum().sort_values(ascending=False).index.values # (O,)
        timestamps = sorted(outcome_df.t.unique()) # (T,)
        # print("Number of outcomes:", len(outcome_set))
        # print("Number of timestamps:", len(timestamps))
        probe_data['t'].append(timestamps)

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
        probe_data['activation'].append(activations_per_t)

        # b. compute entropy
        distribution = []
        for t in timestamps:
            distribution_at_t = []
            for outcome in outcome_set:
                row = outcome_df[(outcome_df['outcome'] == outcome) & (outcome_df['t'] == t)]
                if len(row) == 0:
                    distribution_at_t.append(0.)
                elif len(row) == 1:
                    distribution_at_t.append(row.iloc[0]['outcome_probability'])
                else:
                    assert False, f"Only one row per outcome & t: {outcome} & {t}; #{len(row)}"
            distribution.append(distribution_at_t)

        d = Categorical(probs=torch.tensor(distribution))
        probe_data['entropy'].append(d.entropy())

    del model # done with model once activations are collected!
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
            "num_layers": num_layers
        },
        "metrics": {
            "train_mse": [],
            "test_mse": [],
        },
        "predictions": []
    }

    split_size = int(len(probe_data['entropy']) * test_split) if test_split < 1 else int(test_split)
    for split_index in range(0, len(probe_data['entropy']), split_size):
        activations, entropy = probe_data['activation'], probe_data['entropy']
        activations_test = activations[split_index:split_index + split_size]
        entropy_test = entropy[split_index:split_index + split_size]
        activations_train = activations[:split_index] + activations[split_index + split_size:]
        entropy_train = entropy[:split_index] + entropy[split_index + split_size:] 
        # activations_train, activations_test, entropy_train, entropy_test, ids_train, ids_test = train_test_split(
        #     activations, entropy, question_indices, test_size=test_split, random_state=seed
        # )

        activations_train = torch.cat(activations_train) # (T * num_questions, hidden dim)
        activations_test = torch.cat(activations_test) # (T * num_questions, hidden dim)
        entropy_train = torch.cat(entropy_train).to(activations_train.dtype) # (T * num_questions,)
        entropy_test = torch.cat(entropy_test).to(activations_test.dtype) # (T * num_questions,)
        print("Train inputs:", activations_train.dtype, activations_train.shape)
        print("Train labels:", entropy_train.dtype, entropy_train.shape)
        print("Test inputs:", activations_test.dtype, activations_test.shape)
        print("Test labels:", entropy_test.dtype, entropy_test.shape)

        if probe_class == "linear":
            probe_type = LinearProbe
            probe_kwargs = {}
        elif probe_class == "mlp":
            probe_type = MLPProbe
            probe_kwargs = {
                "hidden_size": hidden_size * activations_train.shape[-1],
                "num_layers": num_layers
            }
        elif probe_class == "attention":
            probe_type = AttentionProbe
            probe_kwargs = {
                "d_proj": 512,
                "nhead": 1,
                "sliding_window": 1024,
            }
        else:
            assert False, f"Probe type {probe_class} not implemented"

        # 3. train probe
        probe = ProbeCV(
            probe_type,
            n_split=cross_val_split,
            input_size=activations_train.shape[-1],
            output_size=1,
            epochs=epochs,
            device="cuda",
            early_stopping=early_stopping,
            patience=patience,
            loss_type="mse",
            learning_rate=learning_rate,
            **probe_kwargs,
        )
        probe.fit(activations_train, entropy_train)

        # 4. evaluate probe on test questions
        train_mse = probe.score(activations_train, entropy_train)
        test_mse = probe.score(activations_test, entropy_test)
        results['metrics']['train_mse'].append(train_mse)
        results['metrics']['test_mse'].append(test_mse)

        # 5. save probe results
        # - save predictions on test questions for side-by-side plot!
        # - repeat for each fold??
        pred_entropy = probe.pred(activations_test) # (T * num_questions,)
        # use ids_test to get question indices
        # use probe data to line up to ts
        # save as json file with:
        # - train mse
        # - test mse
        # - pred entropy #i: []
        t_index = 0
        for question_id in range(split_index, min(split_index + split_size, len(probe_data['entropy']))):
            ts = probe_data['t'][question_id]
            entropy = probe_data['entropy'][question_id]
            pred_entropy_for_q = pred_entropy[t_index:t_index + len(ts)]
            t_index += len(ts) # offset by # of timestamps in each question
            assert len(entropy) == len(ts) and len(entropy) == len(pred_entropy_for_q), f"Ts, true H and pred H must be same length: {len(ts)}, {len(entropy)}, {len(pred_entropy_for_q)}"
            results["predictions"].append({
                "question_id": int(question_id),
                "t": [int(t) for t in ts],
                "true_entropy": [float(e) for e in entropy],
                "pred_entropy": [float(e) for e in pred_entropy_for_q]
            })

    os.makedirs(f"{probe_folder}/{model_nickname}/{dataset_name.lower()}", exist_ok=True)
    # now = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    with open(
        f"{probe_folder}/{model_nickname}/{dataset_name.lower()}/results-{probe_class}-layer{layer}-epochs{epochs}.json",
        "w",
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
    parser.add_argument("--early_stopping", action='store_true')
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=-1)
    parser.add_argument("--num_layers", type=int, default=-1)
    args = parser.parse_args()
    main(**vars(args))
