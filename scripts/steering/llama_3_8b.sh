name=$1
start=$2
end=$3

uv run python steering.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset_name $name \
    --num_paths 500 \
    --temperature 0.7 \
    --max_new_tokens 10000 \
    --layer 20 \
    --token_index -1 \
    --num_outcomes_to_steer 5 \
    --num_steer_samples 10 \
    --batch_size 8 \
    --start_index $start \
    --end_index $end \
    --seed 42