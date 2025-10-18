names=$1

python data_collection.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset_names $names \
    --num_examples 100 \
    --shuffle \
    --num_paths 10 \
    --max_new_tokens 3000 \
    --return_logprobs \
    --return_alternate_texts