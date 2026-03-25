# Circuit Tracer Dashboard

## 1. Overview

Single-page dashboard for visualizing sentence-level attention attribution circuits from the circuit discovery pipeline. It renders a graph where columns are sentences and rows are transformer layers, with edges showing how much each attention head (or layer) attends from one sentence to another. The visualization supports three rendering modes (arcs, flow, and aggregated mask), interactive filtering by threshold and influence percentage, and a detail panel with a Plotly-based metric dropdown covering all evaluation metrics from the roadmap.

## 2. Running the Dashboard

**Via the Python server** (recommended):

```bash
python dashboard/serve.py [port]
```

- Default port is `8765`.
- Serves static files from the `dashboard/` directory.
- Provides a JSON API that auto-discovers mask files matching `results/circuitviz/**/*.json` relative to the project root.
- Open `http://localhost:8765` in a browser.

**Directly in a browser** (no server):

- Open `dashboard/index.html` in a browser.
- Use the "Load Mask" file input to select a JSON file from disk.
- The server dropdown and API endpoints will not be available in this mode.

## 3. Data Format (NodeMask JSON)

The dashboard expects JSON files with the following structure:

```json
{
  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
  "algorithm": "nodewise_attribution",
  "layers": [0, 1, 2, 3],
  "sentences": [
    {"start": 0, "end": 12, "text": "First sentence tokens"},
    {"start": 13, "end": 25, "text": "Second sentence tokens"}
  ],
  "scores": { ... },
  "scores_format": "sparse",
  "metadata": {
    "mask_granularity": "head",
    "threshold_evaluation": [ ... ]
  }
}
```

### Fields

| Field | Description |
|-------|-------------|
| `model_name` | Model identifier string, shown in the sidebar. |
| `algorithm` | Attribution algorithm name. |
| `layers` | Array of integer layer indices included in the mask. |
| `sentences` | Array of `{start, end, text}` objects. `start` and `end` are token indices (inclusive). |
| `scores` | Attribution scores. Structure depends on granularity and format (see below). |
| `scores_format` | Optional. `"sparse"` for `[i, j, score]` triples, absent or `"dense"` for full 2D arrays. |
| `metadata.mask_granularity` | One of `"head"`, `"layer"`, or `"pair"`. |
| `metadata.threshold_evaluation` | Array of evaluation entries with all metrics (see below). |

### Score indexing by granularity

**Dense format** (legacy, default):
- **`head`**: `scores[layer][head][src][tgt]` — per-head 2D array.
- **`layer`**: `scores[layer][src][tgt]` — per-layer 2D array.
- **`pair`**: `scores[src][tgt]` — single 2D array across all layers.

**Sparse format** (`scores_format: "sparse"`):
- Same structure, but each 2D array is replaced with a list of `[i, j, score]` triples for only the active (non-filtered) positions. Filtered positions (diagonal, gap, causal upper triangle) are omitted and reconstructed as zeros on load. The filter is rebuilt from `sentence_gap`, `mask_mode`, and `num_prefix_sentences` in metadata.

### Threshold evaluation entries

Each entry in `metadata.threshold_evaluation` contains metrics computed at a specific threshold. All metrics are computed regardless of which objective was used for mask discovery, whenever the required data (e.g., `answer_ids`) is available.

**Always present:**

| Field | Description |
|-------|-------------|
| `threshold` | Score threshold value. |
| `sparsity` | Fraction of edges ablated at this threshold. |
| `kl_divergence` | Mean per-token KL divergence (clean vs masked), averaged across branches. Plain unweighted. |
| `random_kl_divergence` | Mean of `kl_divergence` across K random baseline masks. |
| `random_kl_divergences` | Array of K individual random baseline values (for std band). |
| `per_sentence_kl` | Per-branch, per-sentence KL. Array of branches, each an array of `{text, mean_kl}` objects. |

**When `branch_rewards` are provided:**

| Field | Description |
|-------|-------------|
| `reward_weighted_kl` | Mean per-token KL weighted by branch reward (+1/-1 for correctness). |
| `random_reward_weighted_kl` | Mean across random baselines. |
| `random_reward_weighted_kls` | Array of K values. |

**When `answer_ids` are provided (IS-based metrics):**

| Field | Description |
|-------|-------------|
| `answer_kl` | KL(P_clean \|\| P_masked) over the answer distribution. P_clean from counting, P_masked via importance sampling. **Objective 1 primary metric.** |
| `reward_gap` | P_masked(target) − max P_masked(other). **Objective 2 primary metric.** |
| `p_target` | P_masked for answer group 0 (target answer). |
| `p_best_other` | max P_masked over non-target answer groups. |
| `answer_probs_masked` | Array of P_masked for each answer group. |
| `n_eff` | Effective sample size from importance weights. |
| `n_eff_ratio` | N_eff / N. Values > 0.1 are considered healthy. |
| `log_weights` | Array of raw log importance weights per chain. |
| `kl_a` | Mean per-token KL over target-answer branches. **Objective 3 metric.** |
| `kl_b` | Mean per-token KL over other-answer branches. **Objective 3 metric.** |
| `contrastive_loss` | `kl_a - kl_b`. Lower = better contrastive circuit. |
| `random_*` | Corresponding random baselines for all IS and contrastive fields. |

**Answer group assignment:** Group 0 ("A") is assigned as follows:
- With `--reward_type correctness`: group 0 = correct answers (reward > 0).
- With `--reward_type cot_length`: group 0 = lowest reward value.
- Default (boxed extraction): group 0 = first unique `\boxed{}` answer encountered.
- Judge fallback: group 0 = first answer cluster from LLM judge.

## 4. Metric Dropdown

The detail panel contains an interactive dropdown for selecting which metric to visualize. All charts are rendered with Plotly.js (zoom, pan, hover tooltips, legend toggle, PNG export). Available metrics depend on which fields exist in the loaded mask:

| Dropdown Label | Field | Chart Type | Description |
|---|---|---|---|
| Per-token KL vs Sparsity | `kl_divergence` | Line + random band | Mean per-token KL at each sparsity level. Always available. |
| Per-sentence KL (across chains) | `per_sentence_kl` | Line (per branch) | One trace per branch, x = sentence index, y = mean KL. Shows where in the chain the mask causes divergence. |
| Per-sentence KL heatmap (by answer) | `per_sentence_kl` + `answer_ids` | Heatmap | Branches on y-axis (grouped by answer), sentences on x-axis, color = KL. Shows A-chains vs B-chains pattern. |
| Answer KL vs Sparsity | `answer_kl` | Line + random band | **Obj 1.** KL between clean and masked answer distributions. Lower = more faithful circuit. |
| Reward Gap vs Sparsity | `reward_gap` | Line + random band | **Obj 2.** P(target) − P(best other). Higher = circuit promotes target answer more. |
| P(target) vs Sparsity | `p_target` | Line + random band | **Obj 2.** Target answer probability under the masked model. |
| P(best other) vs Sparsity | `p_best_other` | Line + random band | **Obj 2.** Best non-target probability. Should degrade faster than random. |
| N_eff / N vs Sparsity | `n_eff_ratio` | Line + threshold + error bars | IS health diagnostic. Red dashed line at 0.1. Error bars from random samples. |
| Per-answer Probabilities | `answer_probs_masked` | Grouped bar | P_clean vs P_masked for each answer group at the selected threshold. |
| Log-weight Histogram | `log_weights` | Histogram | Distribution of log importance weights. Healthy = concentrated near 0. |
| KL_A vs Sparsity | `kl_a` | Line + random band | **Obj 3.** Mean per-token KL for target-answer chains. Should stay low. |
| KL_B vs Sparsity | `kl_b` | Line + random band | **Obj 3.** Mean per-token KL for other-answer chains. Should stay at/above random. |
| Contrastive Loss vs Sparsity | `contrastive_loss` | Line + random band | **Obj 3.** KL_A − KL_B. More negative = better contrastive separation. |

## 5. Architecture

- **Single HTML file** (`index.html`) with inline CSS and JavaScript. No build step required.
- **D3.js v7** for scales, zoom, color interpolation, SVG manipulation.
- **Plotly.js** for interactive metric charts in the detail panel (zoom, pan, hover, legend toggle, export).
- **Dual rendering for the main graph**:
  - **Canvas** for edges/arcs and the grid (performance for hundreds of bezier curves).
  - **SVG** for interactive node circles (native pointer events).
- **Python server** (`serve.py`) provides static file serving plus a JSON API for mask discovery.

## 6. UI Components

### Sidebar Controls

#### Load Mask
File input for local JSON files, plus a dropdown populated from the server API.

#### View Toggle
Three modes:
- **Within-layer arcs**: Bezier arcs within each layer row connecting sentence pairs.
- **Cross-layer flow**: Vertical S-curves grouping edges by (src, tgt) pair across layers.
- **Aggregated mask**: S×S heatmap showing sentence-to-sentence scores aggregated across all active layers.

#### Threshold
Discrete slider over threshold values from `threshold_evaluation` data or auto-generated percentiles.

#### Influence %
Pre-filters edges by cumulative absolute score before threshold is applied.

#### Aggregation
For `head`-granularity masks: combines per-head matrices via mean, max, or sum.

#### Layer Range
Two dropdowns filtering which layers appear. **Disabled for `pair` granularity** (scores are shared across layers, so a note is shown and the graph collapses to a single row).

#### Sentence Legend
Clickable list of sentences. Clicking highlights all connected edges.

### Main Graph

Three rendering modes as described above. Zoom/pan via scroll/drag. Click a node to inspect it in the detail panel.

### Detail Panel

Resizable (drag the left edge, 200–800px range, default 370px). Contains:
- **Metric dropdown**: Interactive Plotly charts for all available metrics.
- **Node Details**: Connection lists for the selected node.

## 7. Server API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/masks` | GET | Returns JSON array of mask file paths. |
| `GET /api/mask?path=<path>` | GET | Returns mask JSON contents. |

## 8. Key Design Decisions

- **Canvas for edges, SVG for nodes.** Performance for many bezier curves vs native pointer events for nodes.
- **Plotly for metric charts.** Provides zoom, pan, hover tooltips, legend toggle, and PNG export without custom implementation.
- **Discrete threshold slider.** Snaps to values from evaluation data for exact lookup.
- **Influence % before threshold.** Prevents low thresholds from flooding the graph.
- **Causal constraint.** Target sentence index ≤ source sentence index (autoregressive attention).
- **All metrics always computed.** When `answer_ids` are available, all three objective metrics are computed regardless of which objective was used for discovery. This enables cross-objective comparison from a single mask.
- **Pair granularity QoL.** Layer controls are disabled since scores don't vary across layers, avoiding user confusion.
