names=$1

python data_collection.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --answer_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset_names $names \
    --num_examples 100 \
    --shuffle \
    --num_paths 10 \
    --max_new_tokens 10000 \
    --return_logprobs \
    --return_alternate_texts