# Usage:
#   bash scripts/forking_paths/llama3_8b_from_collection.sh <prompt_index>
#   bash scripts/forking_paths/llama3_8b_from_collection.sh <prompt_index> <data_path>

export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
unset HF_CACHE_DIR

# Auto-select the first GPU with >75 GB free memory
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 75000 {print $1; exit}')
if [ -z "$FREE_GPU" ]; then
  echo "No GPU with >75 GB free memory available"
  exit 1
fi
echo "Using physical GPU $FREE_GPU"

prompt_index=$1
data_path=${2:-data/collection/deepseek_llama_8b/math_filtered.json}

CUDA_VISIBLE_DEVICES=$FREE_GPU uv run python -m expts.forking_paths.forking_paths_from_collection \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --data_path $data_path \
    --prompt_index $prompt_index \
    --num_branches 32 \
    --max_new_tokens 50000 \
    --temperature 0.6 \
    --seed 42 \
    --enable_prefix_caching
