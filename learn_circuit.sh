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

CUDA_VISIBLE_DEVICES=$FREE_GPU uv run python -m expts.learn_circuit --config expts/configs/answer_kl_patching.yaml

# CUDA_VISIBLE_DEVICES=$FREE_GPU uv run python -m expts.learn_circuit --config expts/configs/thought_anchors.yaml
