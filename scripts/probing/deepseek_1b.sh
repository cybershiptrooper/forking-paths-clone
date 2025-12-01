name=$1

uv run python probing.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset_name $name \
    --layer 20 \
    --test_split 0.1 \
    --cross_val_split 5 \
    --epochs 500 \
    --early_stopping \
    --patience 10 \
    --learning_rate 0.001 \
    --seed 42