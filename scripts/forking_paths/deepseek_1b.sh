name=$1

uv run python forking_paths.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset_name $name \
    --dataset_size 20 \
    --num_branches 20 \
    --max_new_tokens 10000 \
    --temperature 0.7 \
    --seed 42