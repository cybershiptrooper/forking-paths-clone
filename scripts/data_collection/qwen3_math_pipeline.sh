#!/bin/bash
# Pipeline: Data collection + analysis + logic error report for Qwen 3 models on MATH_open
#
# Usage:
#   bash scripts/data_collection/qwen3_math_pipeline.sh          # both 8B and 4B
#   bash scripts/data_collection/qwen3_math_pipeline.sh 8b       # only 8B
#   bash scripts/data_collection/qwen3_math_pipeline.sh 4b       # only 4B

set -e

export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
unset HF_CACHE_DIR

# Collect all GPUs with >75 GB free memory
mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 75000 {print $1}')
if [ ${#FREE_GPUS[@]} -eq 0 ]; then
  echo "No GPU with >75 GB free memory available"
  exit 1
fi
echo "Available GPUs with >75GB free: ${FREE_GPUS[*]}"

WHICH=${1:-both}

run_model() {
    local model_name=$1
    local nickname=$2

    echo ""
    echo "================================================================"
    echo "Step 1: Data collection for $model_name"
    echo "================================================================"

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$FREE_GPU uv run python expts/forking_paths/data_collection_new.py \
        --model_name "$model_name" \
        --dataset_names MATH_open \
        --num_examples 200 \
        --shuffle \
        --num_paths 16 \
        --max_new_tokens 50000 \
        --temperature 0.6 \
        --batch_size 8 \
        --return_logprobs \
        --return_alternate_texts \
        --seed 42 \
        --enable_prefix_caching

    DATA_FILE="data/collection/${nickname}/math_open.json"
    if [ ! -f "$DATA_FILE" ]; then
        echo "ERROR: Expected output $DATA_FILE not found"
        return 1
    fi
    echo "Data collection complete: $DATA_FILE"

    echo ""
    echo "================================================================"
    echo "Step 2: Analysis and filtering for $nickname"
    echo "================================================================"

    FILTERED_FILE="data/collection/${nickname}/math_filtered.json"
    OUTPUT_DIR="results/data_collection_analysis/${nickname}"

    uv run python -m expts.analyse_collected_data \
        --data "$DATA_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --filtered-output "$FILTERED_FILE"

    echo ""
    echo "================================================================"
    echo "Step 3: Logic error report for $nickname"
    echo "================================================================"

    uv run python -c "
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
report_path = output_dir / 'report.json'
if not report_path.exists():
    print('No report found')
    exit(1)

with open(report_path) as f:
    report = json.load(f)

print(f'=== Logic Error Report for $nickname ===')
print(f'Total records: {report[\"summary\"][\"total_records\"]}')
print(f'Filtered (25-75% acc): {report[\"summary\"][\"num_filtered\"]}')
print()

for pid_str, sample in report['filtered_samples'].items():
    wrong = sample.get('wrong_answers', {})
    if not wrong:
        continue
    total_wrong = sum(info['count'] for info in wrong.values())
    logic_count = sum(info['count'] for info in wrong.values() if info['category'] == 'logic_error')
    if logic_count > total_wrong / 2:
        print(f'prompt_id={sample[\"prompt_id\"]}  GT={sample[\"ground_truth\"]}')
        print(f'  {logic_count}/{total_wrong} wrong answers are logic errors')
        for ans, info in wrong.items():
            print(f'    \"{ans}\" x{info[\"count\"]}  cat={info[\"category\"]}  edit_dist={info.get(\"edit_distance\", \"?\")}')
        print()

print('Done.')
"
}

if [ "$WHICH" = "8b" ] || [ "$WHICH" = "both" ]; then
    FREE_GPU=${FREE_GPUS[0]}
    echo "Using GPU $FREE_GPU for Qwen3-8B"
    run_model "Qwen/Qwen3-8B" "qwen3_8b"
fi

if [ "$WHICH" = "4b" ] || [ "$WHICH" = "both" ]; then
    # Use second GPU if available, otherwise re-detect
    if [ ${#FREE_GPUS[@]} -ge 2 ]; then
        FREE_GPU=${FREE_GPUS[1]}
    else
        FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 75000 {print $1; exit}')
    fi
    echo "Using GPU $FREE_GPU for Qwen3-4B"
    run_model "Qwen/Qwen3-4B" "qwen3_4b"
fi

echo ""
echo "================================================================"
echo "Pipeline complete!"
echo "================================================================"
