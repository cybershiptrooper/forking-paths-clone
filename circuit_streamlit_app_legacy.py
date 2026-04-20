"""Interactive Streamlit viewer for NodeMask circuit visualizations."""

from __future__ import annotations

from glob import glob
from pathlib import Path

import streamlit as st

from utils.circuit_streamlit_plotly_legacy import (
    build_attention_pattern_figure,
    build_circuit_graph_figure,
    build_full_circuit_figure,
    build_head_heatmaps_figure,
    build_layer_comparison_figure,
    build_per_sentence_kl_figure,
    build_per_token_kl_figure,
    build_sentence_connection_figure,
    build_sparsity_vs_kl_figure,
    build_threshold_vs_metrics_figure,
    compute_score_range,
)
from utils.masks import NodeMask, load_mask


DEFAULT_MASK_GLOB = "results/circuitviz/**/*.json"


@st.cache_data(show_spinner=False)
def discover_mask_paths(mask_glob: str = DEFAULT_MASK_GLOB) -> list[str]:
    return sorted(glob(mask_glob, recursive=True))


@st.cache_data(show_spinner=False)
def load_node_mask(mask_path: str) -> NodeMask:
    mask = load_mask(mask_path)
    if not isinstance(mask, NodeMask):
        raise ValueError(f"Expected NodeMask, got {type(mask).__name__}")
    return mask


def _format_threshold(value: float) -> str:
    return f"{float(value):.1e}"


def _unique_thresholds(threshold_eval: list[dict]) -> list[float]:
    values = []
    for entry in threshold_eval:
        if isinstance(entry, dict) and "threshold" in entry:
            values.append(float(entry["threshold"]))
    return sorted(set(values))


def _nearest(values: list[float], target: float) -> float:
    return min(values, key=lambda value: abs(value - target))


def _render_threshold_controls(mask: NodeMask, threshold_eval: list[dict]) -> float:
    thresholds = _unique_thresholds(threshold_eval)
    if thresholds:
        default_threshold = _nearest(thresholds, 0.1)
        return st.sidebar.select_slider(
            "Threshold",
            options=thresholds,
            value=default_threshold,
            format_func=_format_threshold,
            help="Global threshold used by threshold-aware plots.",
        )

    min_score, max_score, p99 = compute_score_range(mask)
    upper = max(p99, 0.1)
    if upper <= min_score:
        return st.sidebar.number_input("Threshold", value=0.1, format="%.6e")
    step = max((upper - min_score) / 200.0, 1e-8)
    return st.sidebar.slider(
        "Threshold",
        min_value=float(min_score),
        max_value=float(upper),
        value=float(min(max(0.1, min_score), upper)),
        step=float(step),
        format="%.6e",
        help="Global threshold used by threshold-aware plots.",
    )


def _render_layer_range(all_layers: list[int]) -> tuple[int, int]:
    if len(all_layers) == 1:
        return all_layers[0], all_layers[0]
    return st.sidebar.select_slider(
        "Layer range",
        options=all_layers,
        value=(all_layers[0], all_layers[-1]),
        format_func=lambda value: f"{value}",
        help="Plots aggregate only within this layer range.",
    )


def _filter_layers(all_layers: list[int], layer_start: int, layer_end: int) -> list[int]:
    return [layer for layer in all_layers if layer_start <= layer <= layer_end]


def _render_source_selector() -> str:
    discovered_paths = discover_mask_paths(DEFAULT_MASK_GLOB)
    st.sidebar.subheader("Mask Source")

    selected_path = ""
    if discovered_paths:
        selected_path = st.sidebar.selectbox(
            "Select NodeMask JSON",
            options=discovered_paths,
            index=0,
        )
    manual_path = st.sidebar.text_input(
        "Or enter mask path",
        value="",
        placeholder="results/circuitviz/.../mask.json",
    ).strip()
    return manual_path or selected_path


def main() -> None:
    st.set_page_config(
        layout="wide",
        page_title="Circuit Viewer",
        page_icon="🔬",
    )
    st.title("🔬 Streamlit Circuit Viewer")
    st.caption("Interactive visualization for learned NodeMask circuits.")

    mask_path = _render_source_selector()
    if not mask_path:
        st.info("Select a mask file in the sidebar to begin.")
        st.stop()

    if not Path(mask_path).exists():
        st.error(f"Mask path does not exist: `{mask_path}`")
        st.stop()

    try:
        mask = load_node_mask(mask_path)
    except Exception as exc:  # pragma: no cover - Streamlit runtime reporting
        st.exception(exc)
        st.stop()

    threshold_eval = mask.metadata.get("threshold_evaluation", [])
    if not isinstance(threshold_eval, list):
        threshold_eval = []

    st.sidebar.subheader("Global Filters")
    threshold = _render_threshold_controls(mask, threshold_eval)
    all_layers = sorted(mask.layers)
    layer_start, layer_end = _render_layer_range(all_layers)
    active_layers = _filter_layers(all_layers, layer_start, layer_end)
    skip_k = st.sidebar.number_input(
        "Skip every k layers",
        min_value=0,
        value=0,
        help="When >0, keep every (k+1)-th layer (e.g. 1 → layers 0, 2, 4, …).",
    )
    if skip_k > 0:
        active_layers = active_layers[:: skip_k + 1]
    if not active_layers:
        st.error("No active layers after filtering.")
        st.stop()

    st.sidebar.subheader("Optional Sections")
    circuit_view_mode = st.sidebar.radio(
        "Circuit overview mode",
        options=["Key importance", "Query importance", "Sentence connections"],
        index=0,
        help="Key: which sentences are most attended to. "
        "Query: which sentences attend most to previous ones. "
        "Connections: per-layer arrow diagram.",
    )
    enable_per_layer = st.sidebar.checkbox(
        "Enable per-layer plots",
        value=False,
    )
    per_layer_layers = []
    if enable_per_layer:
        per_layer_layers = st.sidebar.multiselect(
            "Layers for per-layer plots",
            options=active_layers,
            default=active_layers,
        )

    granularity = mask.granularity
    enable_per_head = False
    per_head_layer = active_layers[0]
    selected_heads: list[int] = []
    if granularity == "head":
        enable_per_head = st.sidebar.checkbox(
            "Enable per-head plots",
            value=False,
        )
        if enable_per_head:
            per_head_layer = st.sidebar.selectbox(
                "Layer for per-head plots",
                options=active_layers,
                index=0,
            )
            available_heads = sorted(mask.scores[per_head_layer].keys())
            head_selection_mode = st.sidebar.radio(
                "Head selection mode",
                options=["Top-K", "Manual"],
                index=0,
                horizontal=True,
            )
            if head_selection_mode == "Top-K":
                top_k = st.sidebar.slider(
                    "Top-K heads",
                    min_value=1,
                    max_value=len(available_heads),
                    value=min(5, len(available_heads)),
                )
                ranked = mask.get_head_importance(per_head_layer, threshold=threshold)
                selected_heads = list(ranked.keys())[:top_k]
            else:
                selected_heads = st.sidebar.multiselect(
                    "Select heads",
                    options=available_heads,
                    default=available_heads[: min(5, len(available_heads))],
                )

    st.sidebar.subheader("KL View")
    kl_mode = st.sidebar.radio(
        "KL chart mode",
        options=["Per sentence", "Per token"],
        index=0,
    )

    meta_cols = st.columns(6)
    meta_cols[0].metric("Algorithm", mask.algorithm)
    meta_cols[1].metric("Granularity", granularity)
    meta_cols[2].metric("Layers", len(mask.layers))
    meta_cols[3].metric("Active Layers", len(active_layers))
    meta_cols[4].metric("Sentences", len(mask.sentences))
    meta_cols[5].metric("Threshold", f"{threshold:.1e}")
    st.caption(f"Mask file: `{mask_path}`")

    core_tab, optional_tab, threshold_tab = st.tabs(
        ["Core Plots (Default)", "Optional Per-Layer / Per-Head", "Threshold Eval"]
    )

    with core_tab:
        st.subheader("Layer Comparison")
        comparison_layers = active_layers
        if len(active_layers) > 1:
            cmp_start, cmp_end = st.select_slider(
                "Layer comparison range",
                options=active_layers,
                value=(active_layers[0], active_layers[-1]),
                key="layer_comparison_range",
                help="Limit only the Layer Comparison grid to this range.",
            )
            comparison_layers = _filter_layers(active_layers, cmp_start, cmp_end)
        if not comparison_layers:
            st.warning("No layers selected for Layer Comparison.")
        else:
            st.plotly_chart(
                build_layer_comparison_figure(mask, layers=comparison_layers),
                width="stretch",
            )

        st.subheader("Attention Pattern (Aggregated)")
        st.plotly_chart(
            build_attention_pattern_figure(
                mask,
                layers=active_layers,
                threshold=threshold,
            ),
            width="stretch",
        )

        st.subheader("Circuit Graph (Aggregated)")
        st.plotly_chart(
            build_circuit_graph_figure(
                mask,
                layers=active_layers,
                threshold=threshold,
            ),
            width="stretch",
        )

        st.subheader("Full Circuit Overview")
        if circuit_view_mode == "Sentence connections":
            sent_labels = [f"S{i}" for i in range(len(mask.sentences))]
            highlight_choice = st.selectbox(
                "Highlight sentence",
                options=["None (show all)"] + sent_labels,
                index=0,
                key="highlight_sentence",
                help="Select a sentence to highlight its connections and fade the rest.",
            )
            highlight_idx: int | None = None
            if highlight_choice != "None (show all)":
                highlight_idx = int(highlight_choice[1:])
            st.plotly_chart(
                build_sentence_connection_figure(
                    mask,
                    layers=active_layers,
                    threshold=threshold,
                    highlight_sentence=highlight_idx,
                ),
                width="stretch",
            )
        else:
            mode = "query" if circuit_view_mode == "Query importance" else "key"
            st.plotly_chart(
                build_full_circuit_figure(
                    mask,
                    layers=active_layers,
                    threshold=threshold,
                    mode=mode,
                ),
                width="stretch",
            )

    with optional_tab:
        if enable_per_layer:
            if not per_layer_layers:
                st.warning("Select at least one layer for per-layer plots.")
            else:
                st.subheader("Per-Layer Attention Patterns")
                for layer in per_layer_layers:
                    st.markdown(f"**Layer {layer}**")
                    st.plotly_chart(
                        build_attention_pattern_figure(
                            mask,
                            layers=[layer],
                            threshold=threshold,
                        ),
                        width="stretch",
                    )

                st.subheader("Per-Layer Circuit Graphs")
                for layer in per_layer_layers:
                    st.markdown(f"**Layer {layer}**")
                    st.plotly_chart(
                        build_circuit_graph_figure(
                            mask,
                            layers=[layer],
                            threshold=threshold,
                        ),
                        width="stretch",
                    )
        else:
            st.info("Enable `per-layer plots` in the sidebar to render these sections.")

        if granularity != "head":
            st.info(
                f"Per-head plots are not available for granularity='{granularity}'. "
                "Scores are shared across heads at this granularity."
            )
        elif enable_per_head:
            st.subheader("Per-Head Heatmaps")
            if not selected_heads:
                st.warning("Select at least one head.")
            else:
                st.caption(
                    f"Layer {per_head_layer} | Heads: {', '.join(str(head) for head in selected_heads)}"
                )
                st.plotly_chart(
                    build_head_heatmaps_figure(
                        mask,
                        layer=per_head_layer,
                        heads=selected_heads,
                        threshold=threshold,
                        apply_threshold=True,
                    ),
                    width="stretch",
                )
        else:
            st.info("Enable `per-head plots` in the sidebar to render this section.")

    with threshold_tab:
        if not threshold_eval:
            st.info("No threshold evaluation data found in this mask.")
        else:
            # Build list of available metrics from the threshold_evaluation entries
            first_entry = threshold_eval[0] if threshold_eval else {}
            available_metrics = []

            # Always-available local metrics
            available_metrics.append(("KL Divergence vs Sparsity", "kl_divergence"))
            # Backward compat: old field name
            if any("reward_weighted_objective" in e for e in threshold_eval):
                available_metrics.append(("Reward-Weighted KL vs Sparsity", "reward_weighted_objective"))
            if "reward_weighted_kl" in first_entry:
                available_metrics.append(("Reward-Weighted KL vs Sparsity", "reward_weighted_kl"))
            # IS-based metrics
            if "answer_kl" in first_entry:
                available_metrics.append(("Answer KL vs Sparsity", "answer_kl"))
            # Backward compat: old generic global_metric field
            elif "global_metric" in first_entry:
                available_metrics.append(("Global Metric vs Sparsity", "global_metric"))
            if "reward_gap" in first_entry:
                available_metrics.append(("Reward Gap vs Sparsity", "reward_gap"))
            if "n_eff_ratio" in first_entry:
                available_metrics.append(("N_eff / N vs Sparsity", "n_eff_ratio"))
            # Contrastive metrics
            if "kl_a" in first_entry:
                available_metrics.append(("KL_A (target) vs Sparsity", "kl_a"))
            if "kl_b" in first_entry:
                available_metrics.append(("KL_B (other) vs Sparsity", "kl_b"))
            if "contrastive_loss" in first_entry:
                available_metrics.append(("Contrastive Loss vs Sparsity", "contrastive_loss"))

            metric_labels = [m[0] for m in available_metrics]
            metric_keys = [m[1] for m in available_metrics]
            selected_metric_label = st.radio(
                "Metric",
                metric_labels,
                horizontal=True,
            )
            metric_key = metric_keys[metric_labels.index(selected_metric_label)]

            st.subheader("Threshold vs Metrics")
            threshold_fig = build_threshold_vs_metrics_figure(
                threshold_eval, selected_threshold=threshold, metric_key=metric_key
            )
            if threshold_fig is not None:
                st.plotly_chart(threshold_fig, width="stretch")

            st.subheader("Sparsity vs Metric")
            sparsity_fig = build_sparsity_vs_kl_figure(
                threshold_eval, selected_threshold=threshold, metric_key=metric_key
            )
            if sparsity_fig is not None:
                st.plotly_chart(sparsity_fig, width="stretch")

            st.subheader(f"KL Detail ({kl_mode})")
            if kl_mode == "Per sentence":
                kl_fig = build_per_sentence_kl_figure(threshold_eval, threshold=threshold)
            else:
                kl_fig = build_per_token_kl_figure(threshold_eval, threshold=threshold)

            if kl_fig is None:
                st.info(f"No `{kl_mode}` data found for the selected threshold.")
            else:
                st.plotly_chart(kl_fig, width="stretch")


if __name__ == "__main__":
    main()
