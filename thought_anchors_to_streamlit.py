import argparse
import os
import json
import pandas as pd
from transformers import AutoTokenizer

def search_for_chunk(output_token_ids, chunk_token_ids):
    best_index = -1
    best_overlap = 0
    for i in range(len(output_token_ids)):
        overlap = 0
        for j in range(len(chunk_token_ids)):
            if i + j < len(output_token_ids) and output_token_ids[i + j] == chunk_token_ids[j]:
                overlap += 1
        if overlap > best_overlap:
            best_index = i
            best_overlap = overlap
    return {
        't': best_index,
        'overlap': best_overlap,
        'chunk_length': len(chunk_token_ids)
    }

def main(
    tokenizer_name : str,
    thought_anchors_folder : str,
    output_name : str
):
    with open("config.json") as f:
        config = json.load(f)
        output_folder = config['save_locations']['streamlit_folder']

    output_dir = f"{output_folder}/{output_name}"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    all_base_data = []

    for example_index, problem in enumerate(os.listdir(thought_anchors_folder)):
        # read data from thought anchors
        with open(f"{thought_anchors_folder}/{problem}/base_solution.json") as f:    
            base_solution_data = json.load(f)
        with open(f"{thought_anchors_folder}/{problem}/problem.json") as f:    
            problem_data = json.load(f)        
        with open(f"{thought_anchors_folder}/{problem}/chunks.json") as f:    
            chunk_data = json.load(f)

        # parse base path data
        output_token_ids = tokenizer(chunk_data['solution_text']).input_ids[1:]
        base_data = {
            'clean_answer': base_solution_data['answer'],
            'output_text': chunk_data['solution_text'], 
            'output_token_ids': output_token_ids,
            'question': problem_data['problem'], 
            'correct_letter' : '',
            'correct_answer': problem_data['gt_answer'],
            'all_letters': [], 
            'all_answers': [], 
            'output_logprobs': [],
            'dataset_type': 'open ended'
        }
        all_base_data.append(base_data)
        
        # line up "chunks" with forking path timesteps
        ts = []
        for chunk in chunk_data['chunks']:
            chunk_token_ids = tokenizer(chunk).input_ids[1:]
            t = search_for_chunk(output_token_ids, chunk_token_ids)
            ts.append(t)
        for i in range(len(ts) - 1):
            if abs(ts[i]['t'] + ts[i]['chunk_length'] - ts[i + 1]['t']) > 1:
                print(f"WARNING! Misaligned chunks: {ts[i]['t']} and {ts[i + 1]['t']}")

        # compress forking paths data
        forking_paths_data = []
        for i, t in enumerate(ts):
            with open(f"data/thought_anchors/problem_1591/chunk_{i}/solutions.json") as f:    
                fork_data = json.load(f)

            assert all(fork['chunk_removed'] == fork_data[0]['chunk_removed'] for fork in fork_data)
            assert all(fork['prefix_without_chunk'] == fork_data[0]['prefix_without_chunk'] for fork in fork_data)

            fork_df = pd.DataFrame(fork_data)
            prefix = fork_df.iloc[0]['prefix_without_chunk']
            for outcome in fork_df['answer'].unique():
                outcome_df = fork_df[fork_df['answer'] == outcome]
                sample_rollout = outcome_df.sample(1).iloc[0]['rollout']

                forking_paths_data.append({
                    't': t['t'],
                    'outcome': outcome,
                    'num_rollouts': len(outcome_df),
                    'outcome_probability': len(outcome_df) / len(fork_df),
                    'prefix': prefix,
                    'sample_rollout': sample_rollout
                })

        forking_paths_df = pd.DataFrame(forking_paths_data)
        forking_paths_df.to_csv(f"{output_dir}/{example_index:02d}.csv")
    
    with open(f"{output_dir}/base_data.json", "w+") as f:
        json.dump(all_base_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate forking paths with vLLM")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Model name")
    parser.add_argument("--thought_anchors_folder", type=str, default="", help="Folder from thought anchors to parse")
    parser.add_argument("--output_name", type=str, default="", help="Folder name to save to")

    args = parser.parse_args()
    main(**vars(args))