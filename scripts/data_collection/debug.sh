uv run python data_collection.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset_names AQuA,GSM8k,WildJailBreak \
    --num_examples 2 \
    --shuffle \
    --num_paths 2 \
    --max_new_tokens 400 \
    --return_logprobs \
    --return_alternate_texts