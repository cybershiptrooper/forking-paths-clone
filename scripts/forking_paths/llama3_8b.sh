name=$1

uv run python forking_paths.py \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --dataset_name $name \
    --num_branches 30 \
    --max_new_tokens 16384 \
    --temperature 0.6 \
    --seed 42 \
    --start_index 0 \
    --end_index 50 \
    --enable_prefix_caching