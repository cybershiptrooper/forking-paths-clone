name=$1

uv run python forking_paths.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name $name \
    --num_branches 30 \
    --max_new_tokens 3000 \
    --temperature 0.7 \
    --seed 42 \
    --start_index 0 \
    --end_index 50