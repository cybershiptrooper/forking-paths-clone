names=$1

uv run python data_collection.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_names $names \
    --num_examples 100 \
    --shuffle \
    --num_paths 10 \
    --max_new_tokens 3000 \
    --temperature 0.7 \
    --return_logprobs \
    --return_alternate_texts