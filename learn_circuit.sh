#!/usr/bin/env bash
# Usage: [N_GPUS=N] bash learn_circuit.sh [config.yaml]
# Finds N GPUs with >75 GB free and runs the pipeline on them.
set -e

export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
unset HF_CACHE_DIR

N_GPUS=${N_GPUS:-1}

FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F', ' '$2 > 75000 {print $1}' \
  | head -n "$N_GPUS")

NUM_FOUND=$(echo "$FREE_GPUS" | grep -c .)
if [ "$NUM_FOUND" -lt "$N_GPUS" ]; then
  echo "Only $NUM_FOUND GPU(s) with >75 GB free; need $N_GPUS."
  exit 1
fi

GPU_LIST=$(echo "$FREE_GPUS" | paste -sd, -)
echo "Using GPUs: $GPU_LIST"

CONFIG=${1:-expts/configs/answer_kl_patching.yaml}
echo "Config: $CONFIG"

CUDA_VISIBLE_DEVICES="$GPU_LIST" uv run python -m expts.circuit_discovery.learn_and_evaluate --config "$CONFIG"
