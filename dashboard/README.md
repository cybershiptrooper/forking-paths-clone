# Circuit Tracer Dashboard

## 1. Overview

Single-page dashboard for visualizing sentence-level attention attribution circuits from the circuit discovery pipeline. It renders a graph where columns are sentences and rows are transformer layers, with edges showing how much each attention head (or layer) attends from one sentence to another. The visualization supports two rendering modes (arcs and flow), interactive filtering by threshold and influence percentage, and a detail panel for inspecting individual nodes and the sparsity/KL tradeoff curve.

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
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "algorithm": "integrated_gradients",
  "layers": [0, 1, 2, 3],
  "sentences": [
    {"start": 0, "end": 12, "text": "First sentence tokens"},
    {"start": 13, "end": 25, "text": "Second sentence tokens"}
  ],
  "scores": { ... },
  "metadata": {
    "mask_granularity": "head",
    "threshold_evaluation": [ ... ]
  }
}
```

### Fields

| Field | Description |
|-------|-------------|
| `model_name` | Model identifier string, shown in the sidebar subtitle. |
| `algorithm` | Attribution algorithm name, shown in the sidebar subtitle. |
| `layers` | Array of integer layer indices included in the mask. |
| `sentences` | Array of `{start, end, text}` objects. `start` and `end` are token indices (inclusive). `text` is the decoded sentence content. |
| `scores` | Nested numeric arrays containing attribution scores. Structure depends on granularity (see below). |
| `metadata.mask_granularity` | One of `"head"`, `"layer"`, or `"pair"`. Determines how `scores` is indexed. |
| `metadata.threshold_evaluation` | Optional array of evaluation points (see below). |

### Score indexing by granularity

- **`head`** (most common): `scores[layer][head][src][tgt]` — per-head attribution for each (source sentence, target sentence) pair at each layer.
- **`layer`**: `scores[layer][src][tgt]` — per-layer attribution without head breakdown.
- **`pair`**: `scores[src][tgt]` — a single attribution matrix across all layers.

### Threshold evaluation entries

Each entry in `metadata.threshold_evaluation` has:

```json
{
  "threshold": 0.01,
  "sparsity": 0.85,
  "kl_divergence": 0.12,
  "random_kl_divergence": 1.5,
  "random_kl_divergences": [1.3, 1.5, 1.7],
  "reward_weighted_objective": 0.42
}
```

The `random_kl_divergence` (scalar) or `random_kl_divergences` (array, used for mean/std) provides the baseline for comparison. These points populate the threshold chart in the detail panel.

## 4. Architecture

- **Single HTML file** (`index.html`) with inline CSS and JavaScript. No build step required.
- **D3.js v7** loaded from CDN for scales, zoom behavior, color interpolation, and SVG manipulation.
- **Dual rendering strategy**:
  - **Canvas** for edges/arcs and the grid. Hundreds of bezier curves render much faster on canvas than as SVG path elements.
  - **SVG** for interactive node circles. SVG elements support native pointer events (click, hover, tooltips) without manual hit testing.
- **Python server** (`serve.py`) provides static file serving from the dashboard directory plus a JSON API for discovering and loading mask files.

## 5. UI Components

### Sidebar Controls

#### Load Mask
File input for local JSON files, plus a dropdown populated from the server API. Server masks are fetched via `GET /api/masks` (returns list of paths), then loaded via `GET /api/mask?path=...`.

#### View Toggle
Radio buttons switching between `arcs` and `flow` rendering modes. The current mode is stored in the `viewMode` variable and checked during `render()`.

#### Threshold
Discrete slider over the `thresholdValues` array. If `threshold_evaluation` data is present in the mask metadata, values are taken from there. Otherwise, the dashboard auto-generates percentile values from the score distribution. The current threshold filters out edges with `|score| < threshold`.

#### Influence %
Pre-filters edges by cumulative absolute score before the threshold is applied. All edges are sorted by `|score|` descending, and only the top N% (by cumulative weight) are kept. This prevents the threshold from showing misleading edge counts when many small edges exist.

#### Aggregation
Only available for `head`-granularity masks. Combines per-head score matrices within each layer via `mean`, `max`, or `sum`. Implemented in the `getLayerAgg()` function.

#### Layer Range
Two sliders (min and max) that filter which layers appear in the graph.

#### Sentence Legend
Color-coded list of sentences. Clicking a sentence highlights all edges involving that sentence in the graph.

### Main Graph

#### Layout
Uses `d3.scalePoint` for both axes. The X axis maps sentence indices to horizontal positions; the Y axis maps layer indices to vertical positions (bottom = early layers, top = late layers). Margins are defined in the `MARGIN` constant.

#### Grid
Drawn on canvas via `drawGrid()`. White lines at each layer row and sentence column provide visual reference.

#### Arcs Mode (`drawArcs`)
Bezier curves drawn within each layer row. Arc height scales with the distance between the source and target sentences. Edge styling:
- **Width**: `0.5 + sqrt(norm) * 4`
- **Opacity**: `0.12 + sqrt(norm) * 0.7`
- **Color**: Green (`d3.interpolateGreens`) for positive scores, purple (`d3.interpolatePurples`) for negative scores.
- Small directional arrows are drawn at the curve endpoint.

#### Flow Mode (`drawFlow`)
Answers the question "does a particular sentence-pair connection persist across layers?" Instead of drawing arcs within a single layer row, flow mode groups all edges by their `(src, tgt)` sentence pair into a `pairMap`, then draws vertical S-curve beziers between consecutive layers where that pair has a nonzero score.

**How it works:**
1. All visible edges are bucketed by `src + ',' + tgt` key.
2. Within each bucket, edges are sorted by layer index.
3. For each pair of consecutive layers `(layer_i, layer_{i+1})`, an S-curve bezier is drawn from `(xScale(tgt), yScale(layer_i))` to `(xScale(src), yScale(layer_{i+1}))`. The control points sit at the vertical midpoint, creating a smooth sigmoid-like connector.
4. Small filled circles are drawn at every `(tgt, layer)` and `(src, layer)` position in the group, marking where the connection is active.

**Edge styling** mirrors arcs mode — width and opacity scale with `sqrt(norm)` of the averaged score between the two layer endpoints. Color uses the same green (positive) / purple (negative) scheme.

**When to use it:** Flow mode is most useful for spotting connections that are consistently strong across many layers (they appear as tall, continuous vertical bands) versus connections that only fire at a single layer (isolated dots with no connectors). In arcs mode these cross-layer patterns are invisible because each layer row is drawn independently.

#### Nodes (`drawNodes`)
SVG circles placed at each (sentence, layer) position. Outer ring radius is `5 + normImp * 9` (scaled by node importance). An inner dot shows the importance value. An invisible larger circle provides a generous hit area for clicking. Selected nodes get a magenta dashed border; connected nodes get a green highlight.

#### Zoom and Pan
D3 zoom behavior is attached to the SVG layer. The zoom handler redraws the canvas layer in sync so edges and nodes stay aligned.

### Detail Panel

#### Threshold Chart (`renderThresholdChart`)
Canvas scatter plot in the right panel. X axis = sparsity, Y axis = KL divergence (or reward-weighted objective, toggled by buttons). Points are colored by threshold value using a viridis-like scale. The currently active threshold is marked with a star. The random baseline is shown as an orange dashed horizontal line, with an optional standard deviation band if `random_kl_divergences` is provided. Clicking a point sets that threshold.

#### Node Details (`updateDetail`)
When a node is selected, the detail panel shows:
- **Attends to** (outgoing edges where the node's sentence is the source/query): lists which sentences this node attends to at this layer.
- **Attended by** (incoming edges where the node's sentence is the target/key): lists which sentences attend to this node at this layer.
- Top 20 connections are shown with horizontal score bars.

### Tooltip
Appears on node hover near the cursor. Displays the sentence text, layer number, importance value, and connection counts (outgoing and incoming).

## 6. Data Processing Pipeline

The dashboard processes data through a `boot() -> processData() -> render()` pipeline.

### `boot()`
Called when a mask JSON is loaded. Parses metadata, populates sidebar controls (model name, algorithm, layer range, aggregation options), builds the `thresholdValues` array from threshold evaluation data or auto-generated percentiles, and calls `processData()` followed by `render()`.

### `processData()`
For each layer in the selected range:
1. Gets the aggregated score matrix via `getLayerAgg()` (applies the selected aggregation for head-granularity masks).
2. Collects all edges, enforcing the **causal constraint**: only edges where `tgt <= src` are included (attention can only flow to earlier sentences).
3. Sorts all edges by `|score|` descending.
4. Applies the **influence % cutoff**: keeps the top N% of edges by cumulative absolute score.
5. Applies the **threshold filter**: removes edges with `|score|` below the current threshold.
6. Computes per-node importance as the sum of absolute scores of all edges connected to that node.

### `render()`
Recreates the canvas and SVG elements, then draws in order: grid, edges (arcs or flow), nodes, and axis labels. Sets up the D3 zoom behavior linking canvas and SVG transforms.

### `reprocess()`
Called when any control changes (threshold, influence %, aggregation, layer range, view mode). Re-runs `processData()` and `render()` to update the visualization.

## 7. Server API

The Python server (`serve.py`) exposes two endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/masks` | GET | Returns a JSON array of relative file paths to all mask JSON files matching `results/circuitviz/**/*.json`. |
| `GET /api/mask?path=<relative_path>` | GET | Returns the contents of the specified mask JSON file. |

**Security**: The `_serve_mask` method resolves the requested path and validates that it falls under the project root directory. Requests for paths outside the root receive a `403` response.

All other requests are served as static files from the `dashboard/` directory by the standard `SimpleHTTPRequestHandler`.

## 8. Key Design Decisions

- **Canvas for edges, SVG for nodes.** Hundreds of bezier curves render significantly faster on canvas than as individual SVG path elements. Nodes need pointer events (click, hover), which SVG provides natively without manual hit testing.

- **Discrete threshold slider.** The slider snaps to values from the `threshold_evaluation` data, so each position corresponds to a known sparsity and KL divergence. This allows exact lookup in the threshold chart without interpolation.

- **Influence % applied before threshold.** Sorting by absolute score and keeping only the top N% prevents a low threshold from flooding the graph with many near-zero edges. The threshold then acts as a fine-grained cutoff within the pre-filtered set.

- **Causal constraint (`tgt <= src`).** In autoregressive models, attention can only flow to earlier tokens. Since sentences are ordered, the dashboard enforces that target sentence index is less than or equal to source sentence index, filtering out non-causal edges.

- **Dual-axis point scales.** Using `d3.scalePoint` for both sentence and layer axes makes the layout straightforward and ensures even spacing regardless of the number of sentences or layers.
