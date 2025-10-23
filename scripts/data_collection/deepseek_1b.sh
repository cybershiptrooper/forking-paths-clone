names=$1

uv run python data_collection.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset_names $names \
    --num_examples 100 \
    --shuffle \
    --num_paths 10 \
    --max_new_tokens 10000 \
    --return_logprobs \
    --return_alternate_texts