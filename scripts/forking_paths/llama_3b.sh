name=$1

uv run python forking_paths.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name $name \
    --dataset_size 10 \
    --num_branches 30 \
    --max_new_tokens 3000 \
    --temperature 0.7 \
    --seed 42