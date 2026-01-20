"""Visualization utilities for trajectory graphs."""

from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# Color palette for trajectories
TRAJECTORY_COLORS = (
    px.colors.qualitative.Set3
    + px.colors.qualitative.Pastel1
    + px.colors.qualitative.Dark24
)


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

    def get_fork_points(self) -> list[str]:
        """Get list of available fork points (as string keys)."""
        return sorted(self.fork_points.keys(), key=lambda x: int(x))

    def get_trajectories_for_fork_point(
        self, fork_point: str
    ) -> list[tuple[str, list[tuple[float, int, str]]]]:
        """
        Get all trajectories for a specific fork point.

        Args:
            fork_point: Fork point key (string of character position)

        Returns:
            List of (trajectory_name, trajectory_data) tuples
        """
        if fork_point not in self.fork_points:
            return []

        fork_data = self.fork_points[fork_point]
        trajectories = []
        for key, value in fork_data.items():
            if key.startswith("trajectory_"):
                trajectories.append((key, value))
        return trajectories

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
                    title=dict(text="Visits", side="right"),
                    xanchor="left",
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

    def build_graph_for_fork_point(self, fork_point: str) -> nx.DiGraph:
        """
        Build directed graph for a specific fork point only.

        Args:
            fork_point: Fork point key (string of character position)

        Returns:
            NetworkX DiGraph with only nodes/edges from this fork point's trajectories
        """
        G = nx.DiGraph()

        trajectories = self.get_trajectories_for_fork_point(fork_point)
        if not trajectories:
            return G

        # Collect all nodes and edges from trajectories at this fork point
        all_clusters = set()
        edges = set()

        for traj_name, trajectory in trajectories:
            for i, (t_norm, cluster_id, text) in enumerate(trajectory):
                all_clusters.add(cluster_id)
                if i > 0:
                    prev_cluster = trajectory[i - 1][1]
                    if prev_cluster != cluster_id:
                        edges.add((prev_cluster, cluster_id))

        # Add nodes
        for cluster_id in all_clusters:
            label = self.representative_sentences.get(
                str(cluster_id), f"Cluster {cluster_id}"
            )
            if len(label) > 50:
                label = label[:47] + "..."
            G.add_node(
                cluster_id,
                label=label,
                full_label=self.representative_sentences.get(
                    str(cluster_id), f"Cluster {cluster_id}"
                ),
            )

        # Add edges
        for from_id, to_id in edges:
            G.add_edge(from_id, to_id)

        return G

    def to_plotly_figure_with_paths(
        self,
        fork_point: str,
        layout: str = "kamada_kawai",
        title: str = "Trajectory Paths",
        width: int = 800,
        height: int = 600,
        node_size: int = 30,
        edge_width: int = 2,
        max_t_norm: float = 1.0,
    ) -> go.Figure:
        """
        Create Plotly figure showing individual trajectory paths with different colors.

        - Each trajectory gets a unique color
        - Node and edge sizes are constant
        - Only clusters traversed by trajectories are shown

        Args:
            fork_point: Fork point to visualize
            layout: Layout algorithm ('kamada_kawai', 'spring', 'circular', 'shell')
            title: Figure title
            width: Figure width
            height: Figure height
            node_size: Constant size for all nodes
            edge_width: Constant width for all edges
            max_t_norm: Maximum normalized time to show (for filtering)

        Returns:
            Plotly Figure
        """
        trajectories = self.get_trajectories_for_fork_point(fork_point)
        if not trajectories:
            fig = go.Figure()
            fig.update_layout(
                title=title,
                showlegend=False,
                width=width,
                height=height,
                annotations=[
                    dict(
                        text="No trajectories at this fork point",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                    )
                ],
            )
            return fig

        # Build graph for layout computation
        graph = self.build_graph_for_fork_point(fork_point)

        if len(graph.nodes()) == 0:
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

        traces = []

        # Draw each trajectory as a separate colored path
        for traj_idx, (traj_name, trajectory) in enumerate(trajectories):
            color = TRAJECTORY_COLORS[traj_idx % len(TRAJECTORY_COLORS)]

            # Filter by time if needed
            filtered_traj = [(t, c, txt) for t, c, txt in trajectory if t <= max_t_norm]
            if not filtered_traj:
                continue

            # Get cluster sequence (removing consecutive duplicates for edges)
            cluster_sequence = []
            for t_norm, cluster_id, text in filtered_traj:
                if not cluster_sequence or cluster_sequence[-1] != cluster_id:
                    cluster_sequence.append(cluster_id)

            # Draw edges for this trajectory
            if len(cluster_sequence) > 1:
                edge_x = []
                edge_y = []
                for i in range(len(cluster_sequence) - 1):
                    from_cluster = cluster_sequence[i]
                    to_cluster = cluster_sequence[i + 1]
                    if from_cluster in pos and to_cluster in pos:
                        x0, y0 = pos[from_cluster]
                        x1, y1 = pos[to_cluster]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                if edge_x:
                    edge_trace = go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode="lines",
                        line=dict(width=edge_width, color=color),
                        hoverinfo="none",
                        name=traj_name,
                        showlegend=True,
                        legendgroup=traj_name,
                    )
                    traces.append(edge_trace)

        # Draw nodes (all clusters that appear in any trajectory)
        all_clusters_in_view = set()
        for traj_name, trajectory in trajectories:
            for t_norm, cluster_id, text in trajectory:
                if t_norm <= max_t_norm:
                    all_clusters_in_view.add(cluster_id)

        node_x = []
        node_y = []
        node_text = []
        node_ids = []

        for cluster_id in all_clusters_in_view:
            if cluster_id in pos:
                x, y = pos[cluster_id]
                node_x.append(x)
                node_y.append(y)
                node_ids.append(cluster_id)
                full_label = self.representative_sentences.get(
                    str(cluster_id), f"Cluster {cluster_id}"
                )
                node_text.append(f"Cluster {cluster_id}<br>{full_label[:150]}")

        if node_x:
            # Use a list of constant sizes to ensure uniformity
            constant_sizes = [node_size] * len(node_x)
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                hoverinfo="text",
                text=[str(n) for n in node_ids],
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                hovertext=node_text,
                marker=dict(
                    color="#4a4a4a",
                    size=constant_sizes,
                    sizemode="diameter",
                    line=dict(width=2, color="white"),
                ),
                showlegend=False,
            )
            traces.append(node_trace)

        # Create figure
        fig = go.Figure(data=traces)

        fig.update_layout(
            title=title,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            hovermode="closest",
            width=width,
            height=height,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            margin=dict(l=20, r=150, t=40, b=20),
        )

        return fig
