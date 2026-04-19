# Circuit Discovery

Discover and evaluate sentence-level attention circuits in transformer language models. The pipeline learns which attention heads (or layers, or sentence pairs) are most important for preserving a model's output distribution at a given point in its chain-of-thought, then evaluates the learned mask at multiple sparsity levels.

## Quick start

```bash
uv sync                  # install dependencies
bash learn_circuit.sh    # learn a circuit mask + evaluate it
```

`learn_circuit.sh` auto-selects the first GPU with >75 GB free memory and runs:

```bash
CUDA_VISIBLE_DEVICES=$FREE_GPU uv run python -m expts.circuit_discovery.learn_and_evaluate \
    --config expts/configs/answer_kl_patching.yaml
```

Point it at a different config by editing the `--config` path, or pass CLI args directly (CLI args override config values).

## Pipeline overview

The entry point is `expts/circuit_discovery/learn_and_evaluate.py`, which runs two stages back-to-back:

1. **`learn_circuit.py`** — learns a circuit mask and saves it as a `NodeMask` JSON.
2. **`evaluate_mask.py`** — loads the saved mask, evaluates it at multiple sparsity thresholds, and writes the results back into the same JSON.

### Learning (`learn_circuit.py`)

The learning pipeline has six steps:

1. **Prepare input** — tokenize the prompt and apply the chat template.
2. **Generate branches** — use vLLM to sample a base completion and `num_new_branches` continuations from the `analysis_timestep`. Results are cached to `cache_dir`.
3. **Split into sentences** — segment the token sequence into sentence chunks, optionally including generation-region sentences for `mask_mode=generation|both`.
4. **Group answers** — extract `\boxed{}` answers from branches, cluster them by mathematical equivalence (optionally via an LLM judge), and assign answer IDs for importance-sampling metrics.
5. **Run circuit discovery** — load the model with eager attention, instantiate the chosen algorithm via `create_circuit_discovery()`, and compute per-edge attribution scores.
6. **Save** — write the `NodeMask` (scores, sentences, metadata) to JSON under `output_dir`.

### Evaluation (`evaluate_mask.py`)

Loads a `NodeMask` JSON and its cached completions, then calls `evaluate_at_thresholds()`:

- Converts target sparsity levels (e.g. 0%, 10%, 50%, 90%) into score thresholds.
- At each threshold, zeros out edges below the threshold and measures the resulting KL divergence (and all other available metrics) against the clean model.
- Compares against `num_random_samples` random baseline masks at the same sparsity.
- Writes all results back into the mask JSON under `metadata.threshold_evaluation`.

## Config files

Configs are YAML files in `expts/configs/`. They set default values for any CLI argument; CLI args always override config values.

Example (`answer_kl_patching.yaml`):

```yaml
model_name: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
data_path: data/collection/deepseek_llama_8b/math_open.json
prompt_index: 6
objective: answer_kl
masking_algorithm: nodewise_activation_patching_kv_cache
num_new_branches: 32
mask_granularity: pair
mask_mode: prefix
layers_to_analyse: all
max_sampling_tokens: 10000
analysis_timestep: 1126
sentence_gap: 4
output_dir: results/circuit_discovery/v2
file_name: retain_outcome_dist_at_1126_32_branches
ablate_non_target_layers: true
device: cuda:0
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Model used for vLLM branch generation. |
| `model_to_analyse` | same as `model_name` | Model loaded with eager attention for circuit discovery. |
| `prompt` | built-in math problem | Input prompt. Overridden when using `data_path` + `prompt_index`. |
| `data_path` / `prompt_index` | `None` | Load a question + correct answer from a collection JSON. |
| `masking_algorithm` | `nodewise_attribution` | Circuit discovery algorithm (see below). |
| `objective` | `kl_divergence` | Optimization objective: `kl_divergence`, `log_prob` (local per-token), `answer_kl` (global faithfulness), `reward_gap` (global reward). |
| `mask_granularity` | `head` | Score resolution: `head` (per-head), `layer` (shared across heads), `pair` (shared across layers and heads). |
| `mask_mode` | `prefix` | Which attention region to mask: `prefix` (query=prefix, key=prefix), `generation`, or `both`. |
| `layers_to_analyse` | `[8, 12, 16, 20, 24]` | Layer indices to include, or `all`. |
| `analysis_timestep` | prompt length + 200 | Token index (relative to prompt start) where branches diverge. |
| `num_new_branches` | `8` | Number of continuation branches to sample. |
| `num_ig_steps` | `10` | Integrated gradients interpolation steps. |
| `sentence_gap` | `1` | Minimum sentence index gap for mask pairs. |
| `sentence_chunk` | `1` | Number of sentences to merge into each chunk. |
| `max_sampling_tokens` | `150` | Max tokens for vLLM generation. |
| `num_tokens_to_analyse` | same as `max_sampling_tokens` | Truncate continuations for discovery while keeping full branches for answer extraction. |
| `pair_aggregation` | `mean` | Aggregation over token pairs within a sentence pair: `sum`, `mean`, `median`, `max`. |
| `ablate_non_target_layers` | `false` | Zero out attention in all layers outside `layers_to_analyse`. |
| `renormalize_masked_attention` | `true` | Renormalize post-softmax attention after masking. |
| `reward_type` | `none` | Reward-weighted discovery: `none`, `correctness` (requires `correct_answer`), `cot_length`. |
| `answer_only` | `false` | Restrict the position mask to `\boxed{...}` answer tokens only. |
| `judge_answers` | `false` | Use an LLM judge (via OpenRouter) to cluster branch answers. Falls back automatically if >50% of branches lack `\boxed{}`. |
| `sparsities` | `[0.0, 0.01, ..., 1.0]` | Target sparsity levels for evaluation. |
| `num_random_samples` | `5` | Number of random baseline masks (K) for comparison. |

## Data collection & analysis pipeline

End-to-end pipeline that collects branch samples for a given model on a dataset, judges them against ground truth with an LLM, filters ambiguous samples (25-75% accuracy), and generates per-sample plots + error categorisation.

### One-shot pipeline (recommended)

Use [scripts/data_collection/qwen3_math_pipeline.sh](scripts/data_collection/qwen3_math_pipeline.sh) as the template. It runs data collection → analysis → logic-error report for one or both Qwen 3 models:

```bash
bash scripts/data_collection/qwen3_math_pipeline.sh        # both 8B and 4B
bash scripts/data_collection/qwen3_math_pipeline.sh 8b     # only 8B
bash scripts/data_collection/qwen3_math_pipeline.sh 4b     # only 4B
```

Key params (edit the script to change):

- `--num_examples 200` — number of prompts to sample from the dataset
- `--num_paths 16` — branches per prompt
- `--max_new_tokens 50000` — set high to avoid truncated thinking chains
- `--temperature 0.6`, `--seed 42`

**Always run long-running pipelines in tmux** so they survive SSH disconnects:

```bash
mkdir -p logs
tmux new-session -d -s qwen3_8b "bash scripts/data_collection/qwen3_math_pipeline.sh 8b 2>&1 | tee logs/qwen3_8b_pipeline.log"
tmux attach -t qwen3_8b   # to monitor
```

### Adding a new model

1. Add an entry to `MODEL_METADATA` in [utils/utils.py](utils/utils.py) with `nickname` and `reasoning` fields.
2. Copy `qwen3_math_pipeline.sh` and update the `run_model` calls at the bottom.

### Pipeline steps

**Step 1: Data collection** — [expts/forking_paths/data_collection_new.py](expts/forking_paths/data_collection_new.py) generates `num_paths` samples per prompt for `num_examples` prompts, saves to `data/collection/<model_nickname>/<dataset>.json`. Must be invoked with `PYTHONPATH=.`.

**Step 2: Analysis + filtering** — [expts/analyse_collected_data.py](expts/analyse_collected_data.py) uses an OpenRouter LLM judge (Llama 3.1 8B by default) to score each path, filters samples with 25-75% accuracy, and writes:
- `data/collection/<model_nickname>/math_filtered.json` — full original records of filtered samples
- `results/data_collection_analysis/<model_nickname>/<filtered_index>/` — per-sample plots and `metadata.json` (with `parsed_answers`, `verdicts`, `complete_final_answers`)
- `results/data_collection_analysis/<model_nickname>/report.json` — full summary

The folder name `<filtered_index>` is the array position in `math_filtered.json`, so it plugs directly into `prompt_index` in circuit-discovery configs.

OpenRouter responses are cached in `cache/openrouter/<model>/`. Re-runs of the analysis step are near-instant if prompts haven't changed.

**Step 3: Logic error report** — scans the filtered samples and prints those where >50% of wrong answers are categorised as `logic_error` (vs. `silly_mistake`, `token_error`, `incomplete`).

### Running analysis on existing data

If data was already collected, skip straight to step 2:

```bash
uv run python -m expts.analyse_collected_data \
    --data data/collection/qwen3_8b/math_open.json \
    --output-dir results/data_collection_analysis/qwen3_8b \
    --filtered-output data/collection/qwen3_8b/math_filtered.json
```

### Running forking paths on a filtered prompt

After filtering, run the forking-paths script on a specific prompt using its filtered index:

```bash
bash scripts/forking_paths/llama3_8b_from_collection.sh <filtered_index>
# or with a different data path:
bash scripts/forking_paths/llama3_8b_from_collection.sh <filtered_index> <data_path>
```

This uses [expts/forking_paths/forking_paths_from_collection.py](expts/forking_paths/forking_paths_from_collection.py), which respects the same sentence splitting (`split_tokens_into_sentences`) and sampling params (temp=0.6, seed=42) as `learn_circuit.py` for consistency.

## Circuit discovery algorithms

Algorithms are registered via `utils/circuit_discovery/factory.py`. The available algorithms:

| Algorithm | Method | Description |
|---|---|---|
| `nodewise_attribution` | Integrated Gradients | Interpolates a mask from 0 (fully ablated) to 1 (fully present) and integrates gradients of the objective w.r.t. the mask. Default algorithm. |
| `nodewise_attribution_attention` | AP + IG | Captures clean vs. corrupted attention activations, then applies integrated gradients over the interpolation. Supports richer aggregation (sum, mean, median, max). |
| `nodewise_activation_patching` | Leave-one-out ablation | Zeros each edge individually and measures the resulting change in objective. Forward-pass only (no gradients). |
| `nodewise_activation_patching_kv_cache` | Activation patching + KV cache | Variant that pre-computes the prefix KV cache for efficiency. |
| `nodewise_activation_patching_batch` | Batched activation patching | Variant with configurable `max_batch_size` for throughput. |
| `nodewise_attribution_memory` | Memory-optimized IG | Memory-efficient variant of the integrated gradients approach. |

All algorithms produce a `NodeMask` with the same structure, so evaluation and the dashboard work identically regardless of which algorithm was used.

## Dashboard

The dashboard is a single-page web app (`dashboard/index.html`) for visualizing learned circuit masks.

### Running

```bash
python dashboard/serve.py [port]   # default port 8765
```

Or open `dashboard/index.html` directly in a browser and use the file picker to load a mask JSON (the server API won't be available in this mode).

### Server API

| Endpoint | Description |
|---|---|
| `GET /api/masks` | Lists all mask JSON files matching `results/circuit_discovery/**/*.json`. |
| `GET /api/mask?path=<relative_path>` | Returns the contents of a specific mask JSON. |

### Visualization modes

The main graph renders sentences as columns and layers as rows:

- **Within-layer arcs** — Bezier arcs within each layer row connecting sentence pairs.
- **Cross-layer flow** — Vertical S-curves grouping edges by (src, tgt) pair across layers.
- **Aggregated mask** — S x S heatmap of sentence-to-sentence scores aggregated across all active layers.

### Controls

- **Threshold slider** — snaps to values from evaluation data to filter edges by score.
- **Influence %** — pre-filters edges by cumulative absolute score before threshold is applied.
- **Aggregation** — for `head`-granularity masks, combines per-head matrices via mean, max, or sum.
- **Layer range** — filters which layers appear (disabled for `pair` granularity since scores are shared).
- **Sentence legend** — click a sentence to highlight all its connected edges.

### Detail panel

Contains a metric dropdown with interactive Plotly charts. Available metrics depend on what data is present in the mask:

| Metric | When available | Description |
|---|---|---|
| Per-token KL vs Sparsity | Always | Mean per-token KL at each sparsity level, with random baseline band. |
| Per-sentence KL | Always | Per-branch, per-sentence KL breakdown. |
| Answer KL vs Sparsity | With `answer_ids` | KL between clean and masked answer distributions (Objective 1). |
| Reward Gap vs Sparsity | With `answer_ids` | P(target) - P(best other) (Objective 2). |
| KL_A / KL_B / Contrastive Loss | With `answer_ids` | Per-group KL and contrastive separation (Objective 3). |
| N_eff / N vs Sparsity | With `answer_ids` | Importance sampling health diagnostic. |
| Reward-weighted KL | With `branch_rewards` | KL weighted by branch correctness/length reward. |
