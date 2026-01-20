"""Visualization utilities for trajectory graphs."""

from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np
import plotly.graph_objects as go


class TrajectoryGraphBuilder:
    """Build network graph from trajectories."""

    def __init__(
        self,
        trajectories_data: dict,
        representative_sentences: Optional[dict[str, str]] = None,
    ):
        """
        Initialize with trajectories data.

        Args:
            trajectories_data: Output from TrajectoryBuilder.build_for_prompt()
            representative_sentences: {cluster_id: representative_sentence}
                                      If None, uses from trajectories_data
        """
        self.data = trajectories_data
        self.fork_points = trajectories_data.get("fork_points", {})

        if representative_sentences is not None:
            self.representative_sentences = representative_sentences
        else:
            self.representative_sentences = trajectories_data.get(
                "representative_sentences", {}
            )

        # Pre-compute graph
        self._full_graph: Optional[nx.DiGraph] = None
        self._node_visit_counts: dict[int, int] = {}
        self._edge_weights: dict[tuple[int, int], int] = {}

    def _get_all_trajectories(self) -> list[list[tuple[float, int, str]]]:
        """Get all trajectories as a flat list."""
        all_trajectories = []
        for fork_pos, fork_data in self.fork_points.items():
            for key, value in fork_data.items():
                if key.startswith("trajectory_"):
                    all_trajectories.append(value)
        return all_trajectories

    def build_graph(self) -> nx.DiGraph:
        """
        Build directed graph where:
        - Nodes = cluster IDs
        - Edges = transitions between clusters (weighted by trajectory count)

        Returns:
            NetworkX DiGraph
        """
        if self._full_graph is not None:
            return self._full_graph

        G = nx.DiGraph()

        # Count node visits and edge weights
        node_visits: dict[int, int] = defaultdict(int)
        edge_weights: dict[tuple[int, int], int] = defaultdict(int)

        all_trajectories = self._get_all_trajectories()

        for trajectory in all_trajectories:
            visited_in_traj = set()
            for i, (t_norm, cluster_id, text) in enumerate(trajectory):
                # Count visits
                if cluster_id not in visited_in_traj:
                    node_visits[cluster_id] += 1
                    visited_in_traj.add(cluster_id)

                # Count edges
                if i > 0:
                    prev_cluster = trajectory[i - 1][1]
                    if prev_cluster != cluster_id:  # Don't count self-loops
                        edge_weights[(prev_cluster, cluster_id)] += 1

        # Add nodes
        for cluster_id, count in node_visits.items():
            label = self.representative_sentences.get(str(cluster_id), f"Cluster {cluster_id}")
            # Truncate label for display
            if len(label) > 50:
                label = label[:47] + "..."
            G.add_node(
                cluster_id,
                visits=count,
                label=label,
                full_label=self.representative_sentences.get(str(cluster_id), f"Cluster {cluster_id}"),
            )

        # Add edges
        for (from_id, to_id), weight in edge_weights.items():
            G.add_edge(from_id, to_id, weight=weight)

        self._full_graph = G
        self._node_visit_counts = dict(node_visits)
        self._edge_weights = dict(edge_weights)

        return G

    def filter_by_time(self, max_t_norm: float) -> nx.DiGraph:
        """
        Return subgraph with only transitions up to max_t_norm.

        Args:
            max_t_norm: Maximum normalized time to include

        Returns:
            Filtered NetworkX DiGraph
        """
        G = nx.DiGraph()

        node_visits: dict[int, int] = defaultdict(int)
        edge_weights: dict[tuple[int, int], int] = defaultdict(int)

        all_trajectories = self._get_all_trajectories()

        for trajectory in all_trajectories:
            visited_in_traj = set()
            prev_cluster = None

            for t_norm, cluster_id, text in trajectory:
                if t_norm > max_t_norm:
                    break

                if cluster_id not in visited_in_traj:
                    node_visits[cluster_id] += 1
                    visited_in_traj.add(cluster_id)

                if prev_cluster is not None and prev_cluster != cluster_id:
                    edge_weights[(prev_cluster, cluster_id)] += 1

                prev_cluster = cluster_id

        # Add nodes
        for cluster_id, count in node_visits.items():
            label = self.representative_sentences.get(str(cluster_id), f"Cluster {cluster_id}")
            if len(label) > 50:
                label = label[:47] + "..."
            G.add_node(
                cluster_id,
                visits=count,
                label=label,
                full_label=self.representative_sentences.get(str(cluster_id), f"Cluster {cluster_id}"),
            )

        # Add edges
        for (from_id, to_id), weight in edge_weights.items():
            G.add_edge(from_id, to_id, weight=weight)

        return G

    def to_plotly_figure(
        self,
        graph: Optional[nx.DiGraph] = None,
        layout: str = "kamada_kawai",
        title: str = "Trajectory Graph",
        width: int = 800,
        height: int = 600,
    ) -> go.Figure:
        """
        Convert graph to Plotly figure.
        - Node size ~ number of trajectories passing through
        - Edge width ~ transition count
        - Hover shows representative sentence

        Args:
            graph: NetworkX graph (uses full graph if None)
            layout: Layout algorithm ('kamada_kawai', 'spring', 'circular', 'shell')
            title: Figure title
            width: Figure width
            height: Figure height

        Returns:
            Plotly Figure
        """
        if graph is None:
            graph = self.build_graph()

        if len(graph.nodes()) == 0:
            # Empty graph
            fig = go.Figure()
            fig.update_layout(
                title=title,
                showlegend=False,
                width=width,
                height=height,
                annotations=[
                    dict(
                        text="No data to display",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                    )
                ],
            )
            return fig

        # Compute layout
        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "spring":
            pos = nx.spring_layout(graph, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "shell":
            pos = nx.shell_layout(graph)
        else:
            pos = nx.kamada_kawai_layout(graph)

        # Extract node positions
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            visits = graph.nodes[node].get("visits", 1)
            full_label = graph.nodes[node].get("full_label", f"Cluster {node}")
            node_text.append(f"Cluster {node}<br>Visits: {visits}<br>{full_label[:100]}")
            node_size.append(10 + visits * 2)  # Scale size by visits
            node_color.append(visits)

        # Create edge traces
        edge_traces = []
        max_weight = max((d.get("weight", 1) for _, _, d in graph.edges(data=True)), default=1)

        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get("weight", 1)

            # Normalize weight for width
            width_scaled = 1 + (weight / max_weight) * 4

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width_scaled, color="rgba(150, 150, 150, 0.5)"),
                hoverinfo="none",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[str(n) for n in graph.nodes()],
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale="Viridis",
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title="Visits",
                    xanchor="left",
                    titleside="right",
                ),
                line=dict(width=1, color="white"),
            ),
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            width=width,
            height=height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig

    def get_graph_metrics(self, graph: Optional[nx.DiGraph] = None) -> dict:
        """
        Compute graph metrics.

        Args:
            graph: NetworkX graph (uses full graph if None)

        Returns:
            Dict with graph metrics
        """
        if graph is None:
            graph = self.build_graph()

        if len(graph.nodes()) == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0,
                "avg_degree": 0,
            }

        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
            "num_strongly_connected": nx.number_strongly_connected_components(graph),
            "num_weakly_connected": nx.number_weakly_connected_components(graph),
        }
