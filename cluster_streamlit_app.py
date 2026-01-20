"""Streamlit app for visualizing clustered trajectories."""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cluster_utils.stats import TrajectoryStatistics, get_convergence_info
from cluster_utils.visualization import TrajectoryGraphBuilder
from utils.utils import MODEL_METADATA

# Page config
st.set_page_config(
    layout="wide",
    page_title="Semantic Clustering - Road Not Taken",
    page_icon="🌲",
)

st.title("🌲 Semantic Clustering of Forking Paths")
st.markdown(
    "**Visualize how model reasoning trajectories cluster semantically across different forking points.**"
)

# Load config
with open("config.json") as f:
    config = json.load(f)
streamlit_folder = config["save_locations"]["streamlit_folder"]
clustered_folder = os.path.join("data", "clustered_trajectories")

# Check if clustered data exists
if not os.path.exists(clustered_folder):
    st.error(
        f"No clustered trajectory data found at `{clustered_folder}`. "
        "Please run `cluster_trajectories.py` first."
    )
    st.stop()

# Model selection
available_models = sorted(
    [d for d in os.listdir(clustered_folder) if os.path.isdir(os.path.join(clustered_folder, d))]
)

if not available_models:
    st.error("No models found in clustered trajectories folder.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    model_name = st.selectbox("Select Model:", available_models, index=0)

# Dataset selection
model_dir = os.path.join(clustered_folder, model_name)
available_datasets = sorted(
    [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
)

if not available_datasets:
    st.error(f"No datasets found for model `{model_name}`.")
    st.stop()

with col2:
    dataset_name = st.selectbox("Select Dataset:", available_datasets, index=0)

# Example selection
dataset_dir = os.path.join(model_dir, dataset_name)
available_examples = sorted(
    [f.replace(".json", "") for f in os.listdir(dataset_dir) if f.endswith(".json")]
)

if not available_examples:
    st.error(f"No examples found for dataset `{dataset_name}`.")
    st.stop()

with col3:
    example_id = st.selectbox("Select Example:", available_examples, index=0)

# Load clustered trajectory data
clustered_file = os.path.join(dataset_dir, f"{example_id}.json")
with open(clustered_file) as f:
    traj_data = json.load(f)

# Load base data if available
base_data_file = os.path.join(streamlit_folder, model_name, dataset_name, "base_data.json")
base_data = None
if os.path.exists(base_data_file):
    with open(base_data_file) as f:
        all_base_data = json.load(f)
        idx = int(example_id)
        if idx < len(all_base_data):
            base_data = all_base_data[idx]

# Display question if available
if base_data:
    with st.expander("📋 Question & Answer", expanded=False):
        st.text(base_data.get("question", "Question not available"))
        if "correct_letter" in base_data and "correct_answer" in base_data:
            st.markdown(
                f"**Correct Answer:** {base_data['correct_letter']}) {base_data['correct_answer']}"
            )
        if "clean_answer" in base_data:
            st.markdown(f"**Model Answer:** {base_data['clean_answer']}")

# Metadata
st.sidebar.header("📊 Metadata")
metadata = traj_data.get("metadata", {})
st.sidebar.write(f"**Base Length:** {metadata.get('base_length_chars', 'N/A')} chars")
st.sidebar.write(f"**Num Clusters:** {metadata.get('num_clusters', 'N/A')}")
st.sidebar.write(f"**Embedding Model:** {metadata.get('embedding_model', 'N/A')}")

# Statistics
st.sidebar.header("📈 Statistics")
stats = TrajectoryStatistics(traj_data)
summary = stats.get_summary()

st.sidebar.write(f"**Trajectories:** {summary['num_trajectories']}")
st.sidebar.write(f"**Fork Points:** {summary['num_fork_points']}")
st.sidebar.write(f"**Convergence Rate:** {summary['convergence_rate']:.2%}")
st.sidebar.write(f"**Avg Trajectory Length:** {summary['average_trajectory_length']:.1f}")

# Convergence info
if base_data:
    csv_file = os.path.join(streamlit_folder, model_name, dataset_name, f"{example_id}.csv")
    if os.path.exists(csv_file):
        conv_info = get_convergence_info(
            csv_file, base_data, metadata.get("base_length_chars", 1)
        )
        if conv_info.get("converged_timestep") is not None:
            st.sidebar.header("🎯 Convergence")
            st.sidebar.write(f"**Timestep (norm):** {conv_info['converged_timestep_norm']:.3f}")
            st.sidebar.write(f"**Outcome:** {conv_info['converged_outcome']}")
            is_correct = conv_info.get("is_correct")
            if is_correct is not None:
                st.sidebar.write(f"**Correct:** {'✅' if is_correct else '❌'}")

# Main visualization
st.header("🔗 Trajectory Paths by Fork Point")

# Build graph builder
graph_builder = TrajectoryGraphBuilder(traj_data)

# Fork point selection for visualization
fork_points_list = graph_builder.get_fork_points()
base_length = metadata.get("base_length_chars", 1)

if not fork_points_list:
    st.warning("No fork points found in the data.")
else:
    # Fork point slider
    fork_options_viz = {
        fp: f"t={fp} ({int(fp)/base_length:.1%})" for fp in fork_points_list
    }

    selected_fork_viz = st.select_slider(
        "Select Fork Point to Visualize:",
        options=fork_points_list,
        format_func=lambda x: fork_options_viz[x],
        key="fork_viz_select",
    )

    max_t = st.slider(
        "Maximum normalized time (t_norm):",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Filter trajectories to show only transitions up to this normalized time",
    )

    # Layout and size options
    col_layout, col_width, col_height = st.columns(3)
    with col_layout:
        layout = st.selectbox(
            "Graph Layout:",
            ["kamada_kawai", "spring", "circular", "shell"],
            index=0,
        )
    with col_width:
        fig_width = st.number_input("Width:", value=900, min_value=400, max_value=1600)
    with col_height:
        fig_height = st.number_input("Height:", value=600, min_value=300, max_value=1200)

    # Get trajectories for this fork point
    trajectories_at_fork = graph_builder.get_trajectories_for_fork_point(
        selected_fork_viz
    )
    st.info(
        f"**{len(trajectories_at_fork)} trajectories** at fork point t={selected_fork_viz} ({int(selected_fork_viz)/base_length:.1%})"
    )

    # Create visualization with colored paths
    fig = graph_builder.to_plotly_figure_with_paths(
        fork_point=selected_fork_viz,
        layout=layout,
        title=f"Trajectory Paths at Fork t={selected_fork_viz} ({int(selected_fork_viz)/base_length:.1%})",
        width=int(fig_width),
        height=int(fig_height),
        max_t_norm=max_t,
    )

    st.plotly_chart(fig, width="stretch")

    # Graph metrics for this fork point
    graph = graph_builder.build_graph_for_fork_point(selected_fork_viz)
    metrics = graph_builder.get_graph_metrics(graph)
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    met_col1.metric("Clusters", metrics["num_nodes"])
    met_col2.metric("Transitions", metrics["num_edges"])
    met_col3.metric("Trajectories", len(trajectories_at_fork))
    met_col4.metric("Density", f"{metrics['density']:.3f}")

    # Statistics plots for this fork point
    st.header("📉 Statistics Over Time (for selected fork point)")

    tab1, tab2 = st.tabs(["Graph Width", "Entropy"])

    with tab1:
        width_data = stats.graph_width_over_time_for_fork(selected_fork_viz)
        if width_data:
            fig_width_plot = go.Figure()
            fig_width_plot.add_trace(
                go.Scatter(
                    x=list(width_data.keys()),
                    y=list(width_data.values()),
                    mode="lines+markers",
                    name="Graph Width",
                    line=dict(color="#97B3AE"),
                )
            )
            fig_width_plot.update_layout(
                title=f"Number of Unique Clusters Over Time (Fork t={selected_fork_viz})",
                xaxis_title="Normalized Time",
                yaxis_title="Number of Clusters",
                height=400,
            )
            st.plotly_chart(fig_width_plot, width="stretch")
        else:
            st.info("No data available for this fork point.")

    with tab2:
        entropy_data = stats.trajectory_entropy_over_time_for_fork(selected_fork_viz)
        if entropy_data:
            fig_entropy = go.Figure()
            fig_entropy.add_trace(
                go.Scatter(
                    x=list(entropy_data.keys()),
                    y=list(entropy_data.values()),
                    mode="lines+markers",
                    name="Entropy",
                    line=dict(color="#F2C3B9"),
                )
            )
            fig_entropy.update_layout(
                title=f"Cluster Distribution Entropy Over Time (Fork t={selected_fork_viz})",
                xaxis_title="Normalized Time",
                yaxis_title="Entropy (bits)",
                height=400,
            )
            st.plotly_chart(fig_entropy, width="stretch")
        else:
            st.info("No data available for this fork point.")

# Cluster exploration
st.header("🔍 Explore Clusters")

cluster_sentences = traj_data.get("cluster_sentences", {})
representative_sentences = traj_data.get("representative_sentences", {})

if cluster_sentences:
    cluster_ids = sorted([int(k) for k in cluster_sentences.keys()])
    selected_cluster = st.selectbox(
        "Select Cluster:",
        cluster_ids,
        format_func=lambda x: f"Cluster {x} ({len(cluster_sentences.get(str(x), []))} sentences)",
    )

    if str(selected_cluster) in representative_sentences:
        st.markdown("**Representative Sentence:**")
        st.info(representative_sentences[str(selected_cluster)])

    sentences = cluster_sentences.get(str(selected_cluster), [])
    if sentences:
        st.markdown(f"**All Sentences in Cluster ({len(sentences)}):**")
        # Show unique sentences only
        unique_sentences = list(set(sentences))
        for i, sent in enumerate(unique_sentences[:20]):  # Limit display
            st.text(f"{i+1}. {sent[:200]}{'...' if len(sent) > 200 else ''}")
        if len(unique_sentences) > 20:
            st.text(f"... and {len(unique_sentences) - 20} more unique sentences")

# Fork point exploration
st.header("🍴 Explore Fork Points")

fork_points = traj_data.get("fork_points", {})
if fork_points:
    fork_positions = sorted([int(k) for k in fork_points.keys()])

    # Convert to normalized positions for display
    base_length = metadata.get("base_length_chars", 1)
    fork_options = [
        f"t={int(fp)} ({fp/base_length:.2%})" for fp in fork_positions
    ]

    selected_fork_idx = st.selectbox(
        "Select Fork Point:",
        range(len(fork_positions)),
        format_func=lambda i: fork_options[i],
    )

    selected_fork = str(fork_positions[selected_fork_idx])
    fork_data = fork_points[selected_fork]

    # Display base sentence
    base_sent = fork_data.get("base_sentence", ["", -1])
    st.markdown("**Base Sentence (starting point):**")
    st.info(f"Cluster {base_sent[1]}: {base_sent[0][:300]}{'...' if len(base_sent[0]) > 300 else ''}")

    # Display trajectories
    trajectory_keys = [k for k in fork_data.keys() if k.startswith("trajectory_")]
    st.markdown(f"**Trajectories from this fork point: {len(trajectory_keys)}**")

    if trajectory_keys:
        selected_traj = st.selectbox(
            "Select Trajectory:",
            trajectory_keys,
            format_func=lambda x: f"{x} ({len(fork_data[x])} steps)",
        )

        traj = fork_data[selected_traj]
        st.markdown("**Trajectory Steps:**")

        for i, (t_norm, cluster_id, text) in enumerate(traj[:10]):  # Limit display
            st.markdown(f"**Step {i+1}** (t={t_norm:.3f}, cluster={cluster_id})")
            st.text(text[:300] + ("..." if len(text) > 300 else ""))

        if len(traj) > 10:
            st.text(f"... and {len(traj) - 10} more steps")

# Footer
st.markdown("---")
st.markdown(
    "*Semantic clustering of forking path trajectories. "
    "Nodes represent semantic clusters, edges represent transitions between clusters.*"
)
