name=$1

for layer in 12 16 20 24
do
    uv run python probing.py \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --dataset_name $name \
        --layer $layer \
        --test_split 0.1 \
        --cross_val_split 10 \
        --epochs 500 \
        --early_stopping \
        --patience 10 \
        --learning_rate 0.001 \
        --seed 42
done