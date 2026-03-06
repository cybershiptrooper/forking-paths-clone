"""Benchmark build_sentence_connection_figure.

Run with: uv run python -m tests.test_sentence_connection_perf

Before optimization (one trace + one annotation per arc):
  402 traces, 396 annotations → Avg: 10.85s

After optimization (batched SVG shapes + 2 global traces + per-layer box traces):
  8 traces, 0 annotations → Avg: 0.07s  (~150x speedup)
"""

import time

from utils.masks import load_mask
from utils.circuit_streamlit_plotly import build_sentence_connection_figure

MASK_PATH = (
    "results/circuitviz/ig50_prefix_gap_1/"
    "circuit_nodewise_attribution_layers_all_branches1_ig50.json"
)
LAYERS = [0, 4, 8, 12, 16, 20]
THRESHOLD = 1e-7
N_RUNS = 5


def bench() -> None:
    mask = load_mask(MASK_PATH)
    print(f"Mask: {MASK_PATH}")
    print(f"Layers: {LAYERS}  |  Sentences: {len(mask.sentences)}  |  Threshold: {THRESHOLD}")
    print(f"Runs: {N_RUNS}\n")

    times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        fig = build_sentence_connection_figure(mask, layers=LAYERS, threshold=THRESHOLD)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        n_traces = len(fig.data)
        n_shapes = len(fig.layout.shapes) if fig.layout.shapes else 0
        if i == 0:
            print(f"  Traces: {n_traces}  |  Shapes: {n_shapes}")

    avg = sum(times) / len(times)
    best = min(times)
    print(f"\n  Avg: {avg:.4f}s  |  Best: {best:.4f}s")


if __name__ == "__main__":
    bench()
