#!/bin/bash
# Pipeline: data collection + analysis for Qwen3-8B & Qwen3-14B on MATH and GPQA.
#
# Four parallel jobs across the free GPUs:
#   1. Qwen3-8B  MATH continuation (level>=3, start_index=200, num=500) -> 1 GPU
#   2. Qwen3-8B  GPQA              (gpqa_diamond, num=198)              -> 1 GPU
#   3. Qwen3-14B MATH              (level>=4, num=500)                  -> 2 GPUs (TP=2)
#   4. Qwen3-14B GPQA              (gpqa_diamond, num=198)              -> 2 GPUs (TP=2)
# Total: 6 GPUs.
#
# Usage:
#   bash scripts/data_collection/qwen3_math_gpqa_pipeline.sh           # all 4 jobs
#   bash scripts/data_collection/qwen3_math_gpqa_pipeline.sh 8b_math   # only one
#   JOBS="8b_math 14b_gpqa" bash scripts/data_collection/qwen3_math_gpqa_pipeline.sh
#
# Logs in logs/qwen3_math_gpqa_pipeline_<timestamp>/<job>.log

set -u
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
unset HF_CACHE_DIR

JOBS=${JOBS:-${1:-"8b_math 8b_gpqa 14b_math 14b_gpqa"}}

# Discover GPUs with >75GB free and pre-allocate per job.
mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 75000 {print $1}')
echo "Free GPUs (>75GB): ${FREE_GPUS[*]} (count=${#FREE_GPUS[@]})"

declare -A GPU_FOR
GPU_IDX=0
ALLOCATED_GPUS=""  # set by allocate_gpus, read by caller (avoids subshell)
allocate_gpus() {
    local n=$1
    ALLOCATED_GPUS=""
    for ((k=0; k<n; k++)); do
        if [ $GPU_IDX -ge ${#FREE_GPUS[@]} ]; then
            echo "ERROR: ran out of free GPUs (need $n more for next job)" >&2
            exit 1
        fi
        if [ -z "$ALLOCATED_GPUS" ]; then
            ALLOCATED_GPUS="${FREE_GPUS[$GPU_IDX]}"
        else
            ALLOCATED_GPUS="${ALLOCATED_GPUS},${FREE_GPUS[$GPU_IDX]}"
        fi
        GPU_IDX=$((GPU_IDX + 1))
    done
}

# Decide which jobs are scheduled and how many GPUs each needs.
declare -A NEED_GPUS
for j in $JOBS; do
    case "$j" in
        8b_math|8b_gpqa|14b_math|14b_gpqa) NEED_GPUS[$j]=1 ;;
        *) echo "Unknown job: $j (expected 8b_math, 8b_gpqa, 14b_math, 14b_gpqa)" >&2; exit 1 ;;
    esac
done

# Assign GPUs in the order JOBS lists them. Cannot use $(...) because the
# subshell would discard GPU_IDX increments.
for j in $JOBS; do
    allocate_gpus ${NEED_GPUS[$j]}
    GPU_FOR[$j]=$ALLOCATED_GPUS
done

LOG_DIR="logs/qwen3_math_gpqa_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Log dir: $LOG_DIR"

# ---------------------------------------------------------------------------
# Per-job runner
# ---------------------------------------------------------------------------
run_job() {
    local name=$1
    local model=$2
    local nickname=$3
    local dataset=$4
    local start_idx=$5
    local num=$6
    local min_level=$7
    local suffix=$8
    local outdir=$9
    local filtered=${10}
    local gpus=${11}
    local llm_cat=${12}  # "yes" | "no"

    local tp_size
    tp_size=$(awk -F',' '{print NF}' <<< "$gpus")

    local log="$LOG_DIR/${name}.log"
    {
        echo "=== ${name} ==="
        echo "  GPUs=$gpus (TP=$tp_size)  model=$model  dataset=$dataset"
        echo "  start_index=$start_idx  num=$num  math_min_level=$min_level  suffix='$suffix'"
        echo "  outdir=$outdir  filtered=$filtered  llm_cat=$llm_cat"
        echo "  PID=$$  start=$(date -Is)"
        echo

        # ---- Step 1: collection ----
        echo "--- Step 1: collection ---"
        PYTHONPATH=. CUDA_VISIBLE_DEVICES="$gpus" uv run python expts/forking_paths/data_collection_new.py \
            --model_name "$model" \
            --dataset_names "$dataset" \
            --num_examples "$num" \
            --start_index "$start_idx" \
            --math_min_level "$min_level" \
            --output_suffix "$suffix" \
            --shuffle \
            --num_paths 16 \
            --max_new_tokens 50000 \
            --temperature 0.6 \
            --batch_size 8 \
            --return_logprobs \
            --return_alternate_texts \
            --seed 42 \
            --enable_prefix_caching \
            --tensor_parallel_size "$tp_size"
        local rc=$?
        if [ $rc -ne 0 ]; then
            echo "[$name] collection FAILED (rc=$rc)"; exit $rc
        fi

        # ---- Step 2: analysis ----
        local ds_lower
        ds_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
        local data_file="data/collection/${nickname}/${ds_lower}${suffix}.json"
        if [ ! -f "$data_file" ]; then
            echo "[$name] expected output $data_file not found"; exit 1
        fi

        echo
        echo "--- Step 2: analysis ---"
        local llm_flag=""
        if [ "$llm_cat" = "yes" ]; then llm_flag="--llm_categorise"; fi
        uv run python -m expts.analyse_collected_data \
            --data "$data_file" \
            --output-dir "$outdir" \
            --filtered-output "$filtered" \
            $llm_flag
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "[$name] analysis FAILED (rc=$rc)"; exit $rc
        fi

        echo
        echo "=== ${name}: DONE  end=$(date -Is) ==="
    } > "$log" 2>&1
}

# ---------------------------------------------------------------------------
# Job specs
# ---------------------------------------------------------------------------
spec_8b_math() {
    run_job "8b_math" \
        "Qwen/Qwen3-8B" "qwen3_8b" "MATH_open" \
        200 500 3 "_200_699" \
        "results/data_collection_analysis/qwen3_8b_math_open_200_699" \
        "data/collection/qwen3_8b/math_open_200_699_filtered.json" \
        "${GPU_FOR[8b_math]}" "no"
}
spec_8b_gpqa() {
    run_job "8b_gpqa" \
        "Qwen/Qwen3-8B" "qwen3_8b" "GPQA" \
        0 198 3 "" \
        "results/data_collection_analysis/qwen3_8b_gpqa" \
        "data/collection/qwen3_8b/gpqa_filtered.json" \
        "${GPU_FOR[8b_gpqa]}" "no"
}
spec_14b_math() {
    run_job "14b_math" \
        "Qwen/Qwen3-14B" "qwen3_14b" "MATH_open" \
        0 500 4 "" \
        "results/data_collection_analysis/qwen3_14b" \
        "data/collection/qwen3_14b/math_filtered.json" \
        "${GPU_FOR[14b_math]}" "no"
}
spec_14b_gpqa() {
    run_job "14b_gpqa" \
        "Qwen/Qwen3-14B" "qwen3_14b" "GPQA" \
        0 198 3 "" \
        "results/data_collection_analysis/qwen3_14b_gpqa" \
        "data/collection/qwen3_14b/gpqa_filtered.json" \
        "${GPU_FOR[14b_gpqa]}" "no"
}

# ---------------------------------------------------------------------------
# Launch in parallel
# ---------------------------------------------------------------------------
declare -A PIDS
for j in $JOBS; do
    case "$j" in
        8b_math)   spec_8b_math   & PIDS[$j]=$! ;;
        8b_gpqa)   spec_8b_gpqa   & PIDS[$j]=$! ;;
        14b_math)  spec_14b_math  & PIDS[$j]=$! ;;
        14b_gpqa)  spec_14b_gpqa  & PIDS[$j]=$! ;;
    esac
    echo "Launched $j (GPUs=${GPU_FOR[$j]}) PID=${PIDS[$j]}  log=$LOG_DIR/${j}.log"
done

echo
echo "Waiting for all jobs..."
fail=0
for j in $JOBS; do
    if wait "${PIDS[$j]}"; then
        echo "[$j] OK"
    else
        echo "[$j] FAILED"
        fail=1
    fi
done

if [ $fail -ne 0 ]; then
    echo "One or more jobs failed; see logs in $LOG_DIR"
    exit 1
fi

echo
echo "All jobs done. Logs in $LOG_DIR"
