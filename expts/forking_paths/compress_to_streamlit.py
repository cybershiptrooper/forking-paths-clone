import os
import json
import pandas as pd

def main():
    with open("config.json") as f:
        config = json.load(f)

    forking_paths_folder = config['save_locations']["forking_paths_folder"]
    for model_name in os.listdir(forking_paths_folder):
        for dataset_name in os.listdir(f'{forking_paths_folder}/{model_name}'):
            for forking_paths_file in os.listdir(f'{forking_paths_folder}/{model_name}/{dataset_name}'):
                # load outcome data
                with open(f'{forking_paths_folder}/{model_name}/{dataset_name}/{forking_paths_file}') as f:
                    forking_paths_data = json.load(f)
                
                # compress all rollouts -> counts of outcomes
                df = pd.DataFrame(forking_paths_data)
                if 'clean_answer' not in df.columns.values:
                    print(f"Skipping {model_name}, {dataset_name}, {forking_paths_file}. Missing answer column.")
                    continue

                outcome_df = df.groupby(['t', 'clean_answer'])['norm_cumulative_logprob'].count().reset_index().rename(
                    columns={'norm_cumulative_logprob': 'num_rollouts', 'clean_answer': 'outcome'}
                )
                sum_rollouts = outcome_df.groupby('t')['num_rollouts'].sum().values[0]
                outcome_df['outcome_probability'] = outcome_df['num_rollouts'] / sum_rollouts

                # add example rollout for each timestep/final outcome
                stumps = []
                sample_continuations = []
                for i, row in outcome_df.iterrows():
                    t = row.t
                    outcome = row.outcome

                    random_sample = df[(df['t'] == t) & (df['clean_answer'] == outcome)].sample(1)
                    full_output_text = random_sample['output_text'].values[0]
                    continued_text = random_sample['post_stump_output_text'].values[0]
                    stump = full_output_text[:full_output_text.find(continued_text)]

                    stumps.append(stump)
                    sample_continuations.append(continued_text)
                outcome_df['prefix'] = stumps
                outcome_df['sample_rollout'] = sample_continuations

                os.makedirs(f"{config['save_locations']['streamlit_folder']}/{model_name}/{dataset_name}", exist_ok=True)
                output_file = forking_paths_file.replace('.json', '.csv')
                outcome_df.to_csv(f"{config['save_locations']['streamlit_folder']}/{model_name}/{dataset_name}/{output_file}")


    RELEVANT_KEYS = [
        'clean_answer', 
        'output_text',
        'prompt_token_ids',
        'output_token_ids',
        'question',
        'correct_letter',
        'correct_answer',
        'all_letters',
        'all_answers',
        'output_logprobs',
        'dataset_type'
    ]
    collection_folder = config['save_locations']["collection_folder"]
    for model_name in os.listdir(collection_folder):
        for dataset_file in os.listdir(f'{collection_folder}/{model_name}'):
            with open(f'{collection_folder}/{model_name}/{dataset_file}') as f:
                base_data = json.load(f)
            
            filtered_base_data = [
                {k: base[k] for k in RELEVANT_KEYS}
                for base in base_data
            ]

            dataset_name = dataset_file.replace('.json', '')
            os.makedirs(f"{config['save_locations']['streamlit_folder']}/{model_name}/{dataset_name}", exist_ok=True)
            with open(f"{config['save_locations']['streamlit_folder']}/{model_name}/{dataset_name}/base_data.json", 'w+') as f:
                json.dump(filtered_base_data, f, indent=2)


if __name__ == "__main__":
    main()