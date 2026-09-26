name=$1

uv run python forking_paths.py \
    --model_name microsoft/Phi-4-mini-reasoning \
    --dataset_name $name \
    --dataset_size 50 \
    --num_branches 20 \
    --max_new_tokens 10000 \
    --temperature 0.7 \
    --seed 42