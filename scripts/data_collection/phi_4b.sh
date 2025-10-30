names=$1

uv run python data_collection.py \
    --model_name microsoft/Phi-4-mini-reasoning \
    --dataset_names $names \
    --num_examples 100 \
    --shuffle \
    --num_paths 10 \
    --max_new_tokens 10000 \
    --temperature 0.7 \
    --return_logprobs \
    --return_alternate_texts