"""Plotly utilities for interactive circuit visualization in Streamlit."""

from __future__ import annotations

from math import ceil
import textwrap
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.masks import NodeMask


def truncate_text(text: str, max_chars: int = 24) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def sentence_texts(node_mask: NodeMask) -> list[str]:
    result: list[str] = []
    for idx, sentence in enumerate(node_mask.sentences):
        text = sentence.get("text", "").strip()
        result.append(text if text else f"S{idx}")
    return result


def sentence_labels(node_mask: NodeMask, max_chars: int = 24) -> list[str]:
    texts = sentence_texts(node_mask)
    return [
        f"S{i}: {truncate_text(text, max_chars=max_chars)}"
        for i, text in enumerate(texts)
    ]


def format_hover_sentence(text: str, width: int = 80) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    wrapped = textwrap.wrap(normalized, width=width)
    return "<br>".join(wrapped)


def sentence_hover_texts(node_mask: NodeMask) -> list[str]:
    return [format_hover_sentence(text) for text in sentence_texts(node_mask)]


def _pair_customdata(texts: Sequence[str]) -> np.ndarray:
    num_sents = len(texts)
    customdata = np.empty((num_sents, num_sents, 2), dtype=object)
    for i in range(num_sents):
        for j in range(num_sents):
            customdata[i, j, 0] = texts[i]
            customdata[i, j, 1] = texts[j]
    return customdata


def _aggregate_stack(arrays: list[np.ndarray], aggregation: str = "mean") -> np.ndarray:
    stacked = np.stack(arrays, axis=0)
    if aggregation == "mean":
        return stacked.mean(axis=0)
    if aggregation == "max":
        return stacked.max(axis=0)
    if aggregation == "sum":
        return stacked.sum(axis=0)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def aggregate_selected_layers(
    node_mask: NodeMask,
    layers: Iterable[int],
    aggregation: str = "mean",
) -> np.ndarray:
    layer_list = list(layers)
    if not layer_list:
        raise ValueError("No layers selected.")

    arrays: list[np.ndarray] = []
    for layer in layer_list:
        if layer not in node_mask.scores:
            continue
        for head_scores in node_mask.scores[layer].values():
            arrays.append(np.array(head_scores, dtype=float))

    if not arrays:
        raise ValueError("Selected layers do not contain any head scores.")
    return _aggregate_stack(arrays, aggregation=aggregation)


def compute_score_range(node_mask: NodeMask) -> tuple[float, float, float]:
    all_values = []
    for layer_scores in node_mask.scores.values():
        for head_scores in layer_scores.values():
            all_values.append(np.array(head_scores, dtype=float).ravel())
    flat = np.concatenate(all_values)
    return float(np.min(flat)), float(np.max(flat)), float(np.percentile(flat, 99))


def _single_layer_scores(
    node_mask: NodeMask, layer: int, aggregation: str = "mean"
) -> np.ndarray:
    return np.array(
        node_mask.get_layer_aggregated(layer, aggregation=aggregation), dtype=float
    )


def _apply_causal_triangle(scores: np.ndarray) -> np.ndarray:
    mask = np.tril(np.ones_like(scores, dtype=bool))
    return np.where(mask, scores, np.nan)


def _subplot_dims(num_plots: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(1, num_plots))
    rows = ceil(num_plots / cols)
    return rows, cols


def _safe_vertical_spacing(rows: int, preferred: float = 0.1) -> float:
    if rows <= 1:
        return preferred
    max_allowed = 1.0 / (rows - 1)
    return min(preferred, max_allowed * 0.95)


def build_layer_comparison_figure(
    node_mask: NodeMask,
    layers: Sequence[int],
    aggregation: str = "mean",
) -> go.Figure:
    if not layers:
        raise ValueError("No layers selected.")

    labels = sentence_labels(node_mask)
    customdata = _pair_customdata(sentence_hover_texts(node_mask))
    rows, cols = _subplot_dims(len(layers))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Layer {layer}" for layer in layers],
        horizontal_spacing=0.06,
        vertical_spacing=_safe_vertical_spacing(rows, preferred=0.1),
    )

    for idx, layer in enumerate(layers):
        row = idx // cols + 1
        col = idx % cols + 1
        scores = _single_layer_scores(node_mask, layer, aggregation=aggregation)
        fig.add_trace(
            go.Heatmap(
                z=scores,
                x=labels,
                y=labels,
                customdata=customdata,
                coloraxis="coloraxis",
                xgap=0.5,
                ygap=0.5,
                hovertemplate=(
                    "Query: %{customdata[0]}<br>"
                    "Key: %{customdata[1]}<br>"
                    "Score: %{z:.3e}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            tickangle=-45,
            categoryorder="array",
            categoryarray=labels,
            tickfont=dict(size=12),
            automargin=True,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            autorange="reversed",
            categoryorder="array",
            categoryarray=labels,
            tickfont=dict(size=12),
            automargin=True,
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Layer Comparison ({aggregation} across heads)",
        coloraxis=dict(colorscale="RdYlGn"),
        height=400 * rows,
        plot_bgcolor="lightgray",
        font=dict(size=14),
        hoverlabel=dict(font_size=15, align="left"),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def build_attention_pattern_figure(
    node_mask: NodeMask,
    layers: Sequence[int],
    threshold: float,
    aggregation: str = "mean",
) -> go.Figure:
    scores = aggregate_selected_layers(node_mask, layers, aggregation=aggregation)
    masked = _apply_causal_triangle(scores)
    if threshold > 0:
        masked = np.where(
            np.isnan(masked), np.nan, np.where(masked >= threshold, masked, 0.0)
        )

    labels = sentence_labels(node_mask, max_chars=20)
    customdata = _pair_customdata(sentence_hover_texts(node_mask))
    num_sents = len(labels)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=masked,
                x=labels,
                y=labels,
                customdata=customdata,
                colorscale="Greens",
                xgap=1.0,
                ygap=1.0,
                hovertemplate=(
                    "Query: <br>%{customdata[0]}<br>"
                    "-------------------------------<br>"
                    "Key: <br>%{customdata[1]}<br>"
                    "-------------------------------<br>"
                    "Score: %{z:.3e}<extra></extra>"
                ),
                colorbar=dict(title="Importance"),
            )
        ]
    )
    fig.update_layout(
        title=f"Sentence Pair Importance (Layers {min(layers)}-{max(layers)}; t={threshold:.1e})",
        xaxis_title="Key Sentence",
        yaxis_title="Query Sentence",
        height=max(760, int(32 * num_sents + 300)),
        plot_bgcolor="#f6d5d5",
        font=dict(size=14),
        hovermode="closest",
        hoverlabel=dict(font_size=15, align="left"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(
        tickangle=-45,
        categoryorder="array",
        categoryarray=labels,
        tickfont=dict(size=13),
        automargin=True,
    )
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=labels,
        tickfont=dict(size=13),
        automargin=True,
    )
    return fig


def build_circuit_graph_figure(
    node_mask: NodeMask,
    layers: Sequence[int],
    threshold: float,
    aggregation: str = "mean",
) -> go.Figure:
    scores = aggregate_selected_layers(node_mask, layers, aggregation=aggregation)
    texts = sentence_texts(node_mask)
    hover_texts = sentence_hover_texts(node_mask)
    labels = sentence_labels(node_mask, max_chars=18)
    num_sents = len(texts)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_sents))
    pos = nx.spring_layout(graph, seed=42, k=2.0 / np.sqrt(max(num_sents, 1)))

    red_x: list[float | None] = []
    red_y: list[float | None] = []
    green_segments: list[tuple[float, float, float, float, float]] = []
    mid_x: list[float] = []
    mid_y: list[float] = []
    edge_custom = []

    max_green = (
        float(np.max(scores[scores > threshold])) if np.any(scores > threshold) else 1.0
    )

    for i in range(num_sents):
        for j in range(num_sents):
            if i == j:
                continue
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            score = float(scores[i, j])
            is_important = score > threshold

            if is_important:
                width = 2.0 * score / max_green + 0.5 if max_green > 0 else 1.0
                green_segments.append((x0, y0, x1, y1, width))
            else:
                red_x.extend([x0, x1, None])
                red_y.extend([y0, y1, None])

            mid_x.append((x0 + x1) / 2.0)
            mid_y.append((y0 + y1) / 2.0)
            edge_custom.append(
                (
                    hover_texts[i],
                    hover_texts[j],
                    score,
                    "important" if is_important else "below threshold",
                )
            )

    node_x = [pos[i][0] for i in range(num_sents)]
    node_y = [pos[i][1] for i in range(num_sents)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=red_x,
            y=red_y,
            mode="lines",
            line=dict(color="rgba(217,83,79,0.20)", width=0.8),
            hoverinfo="skip",
            name="Below threshold",
        )
    )
    if green_segments:
        for idx, (x0, y0, x1, y1, width) in enumerate(green_segments):
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color="rgba(46,139,46,0.78)", width=width),
                    hoverinfo="skip",
                    name="Important",
                    showlegend=idx == 0,
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="rgba(46,139,46,0.78)", width=1.5),
                hoverinfo="skip",
                name="Important",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=mid_x,
            y=mid_y,
            mode="markers",
            marker=dict(size=8, color="rgba(0,0,0,0.01)"),
            customdata=edge_custom,
            hovertemplate=(
                "Source: %{customdata[0]}<br>"
                "Target: %{customdata[1]}<br>"
                "Score: %{customdata[2]:.3e}<br>"
                "Status: %{customdata[3]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[f"S{i}" for i in range(num_sents)],
            textposition="top center",
            marker=dict(size=16, color="#A9D6E5", line=dict(width=1, color="#1B4965")),
            customdata=np.array(
                [[labels[i], hover_texts[i]] for i in range(num_sents)], dtype=object
            ),
            hovertemplate="Node: %{customdata[0]}<br>Sentence: %{customdata[1]}<extra></extra>",
            name="Sentences",
        )
    )
    fig.update_layout(
        title=f"Sentence Graph (Layers {min(layers)}-{max(layers)}; t={threshold:.1e})",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=max(760, int(42 * num_sents + 200)),
        font=dict(size=14),
        hoverlabel=dict(font_size=15, align="left"),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h"),
    )
    return fig


def build_full_circuit_figure(
    node_mask: NodeMask,
    layers: Sequence[int],
    threshold: float,
    aggregation: str = "mean",
) -> go.Figure:
    texts = sentence_texts(node_mask)
    hover_texts = sentence_hover_texts(node_mask)
    labels = sentence_labels(node_mask, max_chars=18)
    num_sents = len(texts)
    num_layers = len(layers)
    importance_matrix = np.zeros((num_layers, num_sents), dtype=float)

    for i, layer in enumerate(layers):
        scores = _single_layer_scores(node_mask, layer, aggregation=aggregation)
        causal_mask = np.tril(np.ones_like(scores))
        scores_causal = np.where(causal_mask, scores, 0.0)
        np.fill_diagonal(scores_causal, 0.0)
        col_sum = scores_causal.sum(axis=0)
        valid_counts = np.maximum(causal_mask.sum(axis=0) - 1, 1)
        importance_matrix[i] = col_sum / valid_counts

    customdata = np.empty((num_layers, num_sents, 2), dtype=object)
    for i, layer in enumerate(layers):
        for j, sentence in enumerate(texts):
            customdata[i, j, 0] = layer
            customdata[i, j, 1] = hover_texts[j]

    y_labels = [f"Layer {layer}" for layer in layers]
    fig = go.Figure()

    if threshold > 0:
        binary = (importance_matrix >= threshold).astype(float)
        fig.add_trace(
            go.Heatmap(
                z=binary,
                x=labels,
                y=y_labels,
                text=np.vectorize(lambda x: f"{x:.1e}")(importance_matrix),
                texttemplate="%{text}",
                customdata=customdata,
                colorscale=[
                    [0.0, "#d9534f"],
                    [0.499, "#d9534f"],
                    [0.5, "#5cb85c"],
                    [1.0, "#5cb85c"],
                ],
                showscale=False,
                xgap=1,
                ygap=1,
                hovertemplate=(
                    "Layer: %{customdata[0]}<br>"
                    "Sentence: %{customdata[1]}<br>"
                    "Importance: %{text}<br>"
                    "Status: %{z}<extra></extra>"
                ),
            )
        )
    else:
        fig.add_trace(
            go.Heatmap(
                z=importance_matrix,
                x=labels,
                y=y_labels,
                customdata=customdata,
                colorscale="Greens",
                colorbar=dict(title="Importance"),
                xgap=1,
                ygap=1,
                hovertemplate=(
                    "Layer: %{customdata[0]}<br>"
                    "Sentence: %{customdata[1]}<br>"
                    "Importance: %{z:.3e}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Circuit Overview - Per-Sentence Importance by Layer",
        xaxis_title="Sentence",
        yaxis_title="Layer",
        height=max(620, int(42 * num_layers + 220)),
        plot_bgcolor="lightgray",
        font=dict(size=14),
        hoverlabel=dict(font_size=15, align="left"),
        margin=dict(l=10, r=10, t=60, b=20),
    )
    fig.update_xaxes(
        tickangle=-45,
        categoryorder="array",
        categoryarray=labels,
        tickfont=dict(size=13),
        automargin=True,
    )
    fig.update_yaxes(tickfont=dict(size=13), automargin=True)
    return fig


def _valid_threshold_entries(threshold_eval: Sequence[dict]) -> list[dict]:
    return [
        entry
        for entry in threshold_eval
        if isinstance(entry, dict) and "threshold" in entry
    ]


def nearest_threshold_entry(
    threshold_eval: Sequence[dict], threshold: float
) -> dict | None:
    entries = _valid_threshold_entries(threshold_eval)
    if not entries:
        return None
    return min(
        entries, key=lambda entry: abs(float(entry["threshold"]) - float(threshold))
    )


def build_threshold_vs_metrics_figure(
    threshold_eval: Sequence[dict],
    selected_threshold: float,
) -> go.Figure | None:
    entries = _valid_threshold_entries(threshold_eval)
    if not entries:
        return None

    thresholds = np.array([float(entry["threshold"]) for entry in entries], dtype=float)
    sparsities = np.array(
        [float(entry.get("sparsity", np.nan)) for entry in entries], dtype=float
    )
    kl_scores = np.array(
        [float(entry.get("kl_divergence", np.nan)) for entry in entries], dtype=float
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=thresholds, y=sparsities, mode="lines+markers", name="Sparsity"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=thresholds, y=kl_scores, mode="lines+markers", name="KL Divergence"),
        secondary_y=True,
    )
    fig.add_vline(x=float(selected_threshold), line_dash="dash", line_color="#444")
    fig.update_xaxes(title_text="Threshold", tickformat=".1e")
    fig.update_yaxes(title_text="Sparsity", secondary_y=False, tickformat=".1%")
    fig.update_yaxes(title_text="KL Divergence", secondary_y=True)
    fig.update_layout(
        title="Threshold vs Sparsity/KL Divergence",
        height=520,
        font=dict(size=14),
        hoverlabel=dict(font_size=15, align="left"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_sparsity_vs_kl_figure(
    threshold_eval: Sequence[dict],
    selected_threshold: float,
) -> go.Figure | None:
    entries = _valid_threshold_entries(threshold_eval)
    if not entries:
        return None

    thresholds = np.array([float(entry["threshold"]) for entry in entries], dtype=float)
    sparsities = np.array(
        [float(entry.get("sparsity", np.nan)) for entry in entries], dtype=float
    )
    kl_scores = np.array(
        [float(entry.get("kl_divergence", np.nan)) for entry in entries], dtype=float
    )

    sort_idx = np.argsort(sparsities)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sparsities[sort_idx],
            y=kl_scores[sort_idx],
            mode="lines",
            line=dict(color="gray", width=1),
            name="Trend",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sparsities,
            y=kl_scores,
            mode="markers",
            marker=dict(
                size=8,
                color=thresholds,
                colorscale="Viridis",
                colorbar=dict(title="Threshold"),
                line=dict(width=0.5, color="black"),
            ),
            customdata=np.stack([thresholds], axis=1),
            hovertemplate=(
                "Sparsity: %{x:.2%}<br>"
                "KL: %{y:.3e}<br>"
                "Threshold: %{customdata[0]:.1e}<extra></extra>"
            ),
            name="Threshold points",
        )
    )

    # Check for per-sample random KLs (K random masks)
    has_multi_random = all(
        "random_kl_divergences" in entry
        and isinstance(entry["random_kl_divergences"], list)
        and len(entry["random_kl_divergences"]) > 0
        for entry in entries
    )
    if has_multi_random:
        random_all = np.array(
            [entry["random_kl_divergences"] for entry in entries], dtype=float
        )  # shape (num_thresholds, K)
        random_mean = random_all.mean(axis=1)
        random_std = random_all.std(axis=1)
        random_mean_sorted = random_mean[sort_idx]
        random_std_sorted = random_std[sort_idx]
        sparsities_sorted = sparsities[sort_idx]

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=sparsities_sorted,
                y=random_mean_sorted,
                mode="lines+markers",
                line=dict(color="orange", dash="dash"),
                marker=dict(symbol="x"),
                name="Random baseline (mean)",
            )
        )
        # Error band (mean +/- 1 std)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([sparsities_sorted, sparsities_sorted[::-1]]),
                y=np.concatenate([
                    random_mean_sorted + random_std_sorted,
                    (random_mean_sorted - random_std_sorted)[::-1],
                ]),
                fill="toself",
                fillcolor="rgba(255, 165, 0, 0.2)",
                line=dict(color="rgba(255, 165, 0, 0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Random baseline (\u00b11\u03c3)",
            )
        )
    elif all(
        "random_kl_divergence" in entry and entry["random_kl_divergence"] is not None
        for entry in entries
    ):
        random_scores = np.array(
            [float(entry["random_kl_divergence"]) for entry in entries], dtype=float
        )
        fig.add_trace(
            go.Scatter(
                x=sparsities[sort_idx],
                y=random_scores[sort_idx],
                mode="lines+markers",
                line=dict(color="orange", dash="dash"),
                marker=dict(symbol="x"),
                name="Random baseline",
            )
        )

    selected_idx = int(np.argmin(np.abs(thresholds - float(selected_threshold))))
    fig.add_trace(
        go.Scatter(
            x=[sparsities[selected_idx]],
            y=[kl_scores[selected_idx]],
            mode="markers",
            marker=dict(size=14, symbol="star", color="black"),
            name="Selected threshold",
        )
    )
    fig.update_layout(
        title="Sparsity vs KL Divergence",
        xaxis_title="Sparsity",
        yaxis_title="KL Divergence",
        height=520,
        font=dict(size=14),
        hoverlabel=dict(font_size=15, align="left"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(tickformat=".0%")
    return fig


def build_per_token_kl_figure(
    threshold_eval: Sequence[dict], threshold: float
) -> go.Figure | None:
    entry = nearest_threshold_entry(threshold_eval, threshold)
    if entry is None or "per_token_kl" not in entry:
        return None

    branches = entry["per_token_kl"]
    fig = go.Figure()
    for branch_idx, branch_kl in enumerate(branches):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(branch_kl))),
                y=branch_kl,
                mode="lines",
                name=f"Branch {branch_idx}",
                hovertemplate=(
                    f"Branch: {branch_idx}<br>"
                    "Token index: %{x}<br>"
                    "KL: %{y:.3e}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=(
            f"Per-Token KL Across Branches "
            f"(threshold={float(entry['threshold']):.1e}, "
            f"sparsity={float(entry.get('sparsity', 0.0)):.1%})"
        ),
        xaxis_title="Continuation Token Index",
        yaxis_title="KL Divergence",
        height=520,
        font=dict(size=14),
        hoverlabel=dict(font_size=15, align="left"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_per_sentence_kl_figure(
    threshold_eval: Sequence[dict], threshold: float
) -> go.Figure | None:
    entry = nearest_threshold_entry(threshold_eval, threshold)
    if entry is None or "per_sentence_kl" not in entry:
        return None

    branches = entry["per_sentence_kl"]
    fig = go.Figure()
    for branch_idx, branch_sentences in enumerate(branches):
        values = [float(sentence.get("mean_kl", 0.0)) for sentence in branch_sentences]
        texts = [sentence.get("text", "") for sentence in branch_sentences]
        customdata = np.array(texts, dtype=object)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode="lines+markers",
                marker=dict(symbol="x"),
                customdata=customdata,
                name=f"Branch {branch_idx}",
                hovertemplate=(
                    f"Branch: {branch_idx}<br>"
                    "Sentence index: %{x}<br>"
                    "Sentence: %{customdata}<br>"
                    "Mean KL: %{y:.3e}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=(
            f"Per-Sentence KL Across Branches "
            f"(threshold={float(entry['threshold']):.1e}, "
            f"sparsity={float(entry.get('sparsity', 0.0)):.1%})"
        ),
        xaxis_title="Continuation Sentence Index",
        yaxis_title="Mean KL Divergence",
        height=520,
        font=dict(size=14),
        hoverlabel=dict(font_size=15, align="left"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_head_heatmaps_figure(
    node_mask: NodeMask,
    layer: int,
    heads: Sequence[int],
    threshold: float,
    apply_threshold: bool = True,
) -> go.Figure:
    if not heads:
        raise ValueError("No heads selected.")

    labels = sentence_labels(node_mask, max_chars=18)
    customdata = _pair_customdata(sentence_hover_texts(node_mask))
    rows, cols = _subplot_dims(len(heads))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Head {head}" for head in heads],
        horizontal_spacing=0.06,
        vertical_spacing=_safe_vertical_spacing(rows, preferred=0.1),
    )

    for idx, head in enumerate(heads):
        row = idx // cols + 1
        col = idx % cols + 1
        scores = np.array(node_mask.scores[layer][head], dtype=float)
        scores = _apply_causal_triangle(scores)
        if apply_threshold and threshold > 0:
            scores = np.where(
                np.isnan(scores), np.nan, np.where(scores >= threshold, scores, 0.0)
            )

        fig.add_trace(
            go.Heatmap(
                z=scores,
                x=labels,
                y=labels,
                customdata=customdata,
                coloraxis="coloraxis",
                xgap=1,
                ygap=1,
                hovertemplate=(
                    "Query: %{customdata[0]}<br>"
                    "Key: %{customdata[1]}<br>"
                    "Score: %{z:.3e}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            tickangle=-45,
            categoryorder="array",
            categoryarray=labels,
            tickfont=dict(size=12),
            automargin=True,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            autorange="reversed",
            categoryorder="array",
            categoryarray=labels,
            tickfont=dict(size=12),
            automargin=True,
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Head Heatmaps (Layer {layer})",
        coloraxis=dict(colorscale="RdYlGn"),
        height=400 * rows,
        plot_bgcolor="black",
        font=dict(size=14),
        hoverlabel=dict(font_size=15, align="left"),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig
