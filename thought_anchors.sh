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

CONFIG=${1:-expts/configs/answer_kl_patching.yaml}
echo "Config: $CONFIG"

# Step 1: Compute attention suppression scores
echo "=== Computing suppression scores ==="
CUDA_VISIBLE_DEVICES=$FREE_GPU uv run python -m expts.thought_anchor_analysis --config "$CONFIG"

# Step 2: Evaluate the resulting mask
# Derive the output path from the config's file_name (defaults to the suppression naming)
MASK_PATH=$(PYTHONPATH=. .venv/bin/python -c "
from utils.expt_config import load_config
import os
c = load_config('$CONFIG')
output_dir = c.get('output_dir', 'results/circuit_discovery')
fn = c.get('file_name')
if fn:
    base = fn.removesuffix('.json')
    print(os.path.join(output_dir, f'{base}_attention_suppression.json'))
else:
    nb = c.get('num_new_branches', 8)
    print(os.path.join(output_dir, f'attention_suppression_branches{nb}.json'))
")
echo "=== Evaluating mask at $MASK_PATH ==="
CUDA_VISIBLE_DEVICES=$FREE_GPU uv run python -m expts.circuit_discovery.evaluate_mask --mask_path "$MASK_PATH"