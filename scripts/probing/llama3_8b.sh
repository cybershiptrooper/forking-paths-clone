name=$1
probe=$2

for layer in 12
do
    uv run python probing_classifier.py \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --dataset_name $name \
        --probe_class $probe \
        --layer $layer \
        --test_split 0.2 \
        --cross_val_split 10 \
        --epochs 50 \
        --early_stopping \
        --patience 25 \
        --learning_rate 0.001 \
        --seed 42 \
        --hidden_size 2 \
        --num_layers 3 
done