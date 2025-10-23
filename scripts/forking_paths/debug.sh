uv run python forking_paths.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name AQuA \
    --dataset_size 2 \
    --num_branches 10 \
    --max_new_tokens 400 \
    --temperature 0.7 \
    --seed 42