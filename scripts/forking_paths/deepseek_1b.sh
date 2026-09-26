name=$1

uv run python forking_paths.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset_name $name \
    --num_branches 20 \
    --max_new_tokens 10000 \
    --temperature 0.7 \
    --seed 42 \
    --start_index 0 \
    --end_index 20