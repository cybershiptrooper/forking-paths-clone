names=$1

uv run python data_collection.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset_names $names \
    --num_examples 500 \
    --shuffle \
    --num_paths 20 \
    --max_new_tokens 16384 \
    --temperature 0.7 \
    --return_logprobs \
    --return_alternate_texts \
    --seed 50 \
    --enable_prefix_caching