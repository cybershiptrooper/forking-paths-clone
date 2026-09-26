"""Statistical analysis for trajectory sets."""

import json
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

# Import from parent directory
sys.path.insert(0, "..")
from utils.cot_analysis import get_final_convergence, load_data


class TrajectoryStatistics:
    """Compute statistics over trajectory sets."""

    def __init__(self, trajectories_data: dict):
        """
        Initialize with trajectories data.

        Args:
            trajectories_data: Output from TrajectoryBuilder.build_for_prompt()
                               Contains 'fork_points', 'metadata', etc.
        """
        self.data = trajectories_data
        self.fork_points = trajectories_data.get("fork_points", {})
        self.metadata = trajectories_data.get("metadata", {})

    def _get_all_trajectories(self) -> list[list[tuple[float, int, str]]]:
        """Get all trajectories as a flat list."""
        all_trajectories = []
        for fork_pos, fork_data in self.fork_points.items():
            for key, value in fork_data.items():
                if key.startswith("trajectory_"):
                    all_trajectories.append(value)
        return all_trajectories

    def _get_trajectories_for_fork_point(
        self, fork_point: str
    ) -> list[list[tuple[float, int, str]]]:
        """Get trajectories for a specific fork point."""
        if fork_point not in self.fork_points:
            return []

        fork_data = self.fork_points[fork_point]
        trajectories = []
        for key, value in fork_data.items():
            if key.startswith("trajectory_"):
                trajectories.append(value)
        return trajectories

    def graph_width_over_time_for_fork(
        self, fork_point: str, num_bins: int = 20
    ) -> dict[float, int]:
        """
        Compute number of unique clusters at each normalized time bin for a specific fork point.

        Args:
            fork_point: Fork point key (string of character position)
            num_bins: Number of time bins

        Returns:
            {t_norm_bin: num_unique_clusters, ...}
        """
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_clusters: dict[int, set[int]] = {i: set() for i in range(num_bins)}

        trajectories = self._get_trajectories_for_fork_point(fork_point)

        for trajectory in trajectories:
            for t_norm, cluster_id, _ in trajectory:
                bin_idx = min(int(t_norm * num_bins), num_bins - 1)
                bin_clusters[bin_idx].add(cluster_id)

        return {float(bin_centers[i]): len(bin_clusters[i]) for i in range(num_bins)}

    def trajectory_entropy_over_time_for_fork(
        self, fork_point: str, num_bins: int = 20
    ) -> dict[float, float]:
        """
        Compute entropy of cluster distribution at each time bin for a specific fork point.

        Args:
            fork_point: Fork point key (string of character position)
            num_bins: Number of time bins

        Returns:
            {t_norm_bin: entropy, ...}
        """
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_cluster_counts: dict[int, dict[int, int]] = {
            i: defaultdict(int) for i in range(num_bins)
        }

        trajectories = self._get_trajectories_for_fork_point(fork_point)

        for trajectory in trajectories:
            for t_norm, cluster_id, _ in trajectory:
                bin_idx = min(int(t_norm * num_bins), num_bins - 1)
                bin_cluster_counts[bin_idx][cluster_id] += 1

        result = {}
        for i in range(num_bins):
            counts = list(bin_cluster_counts[i].values())
            if counts:
                total = sum(counts)
                probs = [c / total for c in counts]
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            else:
                entropy = 0.0
            result[float(bin_centers[i])] = entropy

        return result

    def graph_width_over_time(self, num_bins: int = 20) -> dict[float, int]:
        """
        Compute number of unique clusters at each normalized time bin.

        Args:
            num_bins: Number of time bins

        Returns:
            {t_norm_bin: num_unique_clusters, ...}
        """
        # Create time bins
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Collect clusters for each bin
        bin_clusters: dict[int, set[int]] = {i: set() for i in range(num_bins)}

        all_trajectories = self._get_all_trajectories()

        for trajectory in all_trajectories:
            for t_norm, cluster_id, _ in trajectory:
                # Find which bin this belongs to
                bin_idx = min(int(t_norm * num_bins), num_bins - 1)
                bin_clusters[bin_idx].add(cluster_id)

        return {
            float(bin_centers[i]): len(bin_clusters[i])
            for i in range(num_bins)
        }

    def trajectory_entropy_over_time(self, num_bins: int = 20) -> dict[float, float]:
        """
        Compute entropy of cluster distribution at each time bin.
        Higher entropy = more diverse paths.

        Args:
            num_bins: Number of time bins

        Returns:
            {t_norm_bin: entropy, ...}
        """
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Count clusters for each bin
        bin_cluster_counts: dict[int, dict[int, int]] = {
            i: defaultdict(int) for i in range(num_bins)
        }

        all_trajectories = self._get_all_trajectories()

        for trajectory in all_trajectories:
            for t_norm, cluster_id, _ in trajectory:
                bin_idx = min(int(t_norm * num_bins), num_bins - 1)
                bin_cluster_counts[bin_idx][cluster_id] += 1

        # Compute entropy for each bin
        result = {}
        for i in range(num_bins):
            counts = list(bin_cluster_counts[i].values())
            if counts:
                total = sum(counts)
                probs = [c / total for c in counts]
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            else:
                entropy = 0.0
            result[float(bin_centers[i])] = entropy

        return result

    def convergence_rate(self) -> float:
        """
        Fraction of trajectories that end in the same cluster as base trajectory.

        Returns:
            Convergence rate between 0 and 1
        """
        all_trajectories = self._get_all_trajectories()
        if not all_trajectories:
            return 0.0

        # Get base trajectory final cluster for each fork point
        base_final_clusters = {}
        for fork_pos, fork_data in self.fork_points.items():
            base_sent = fork_data.get("base_sentence", (None, -1))
            # base_sentence is (text, cluster_id)
            base_final_clusters[fork_pos] = base_sent[1]

        # Count how many trajectories end in same cluster as base
        converged = 0
        total = 0

        for fork_pos, fork_data in self.fork_points.items():
            base_cluster = base_final_clusters.get(fork_pos, -1)
            for key, trajectory in fork_data.items():
                if key.startswith("trajectory_"):
                    if trajectory:  # Non-empty trajectory
                        final_cluster = trajectory[-1][1]  # Last entry's cluster
                        if final_cluster == base_cluster:
                            converged += 1
                    total += 1

        return converged / total if total > 0 else 0.0

    def branching_factor(self) -> dict[int, float]:
        """
        For each cluster, compute average number of distinct next-clusters.

        Returns:
            {cluster_id: avg_branching_factor}
        """
        # Track transitions from each cluster
        transitions: dict[int, set[int]] = defaultdict(set)

        all_trajectories = self._get_all_trajectories()

        for trajectory in all_trajectories:
            for i in range(len(trajectory) - 1):
                from_cluster = trajectory[i][1]
                to_cluster = trajectory[i + 1][1]
                transitions[from_cluster].add(to_cluster)

        return {
            cluster_id: float(len(next_clusters))
            for cluster_id, next_clusters in transitions.items()
        }

    def average_trajectory_length(self) -> float:
        """
        Compute average number of cluster transitions per trajectory.

        Returns:
            Average trajectory length
        """
        all_trajectories = self._get_all_trajectories()
        if not all_trajectories:
            return 0.0

        lengths = [len(t) for t in all_trajectories]
        return np.mean(lengths)

    def cluster_visit_counts(self) -> dict[int, int]:
        """
        Count how many trajectories visit each cluster.

        Returns:
            {cluster_id: visit_count}
        """
        counts: dict[int, int] = defaultdict(int)

        all_trajectories = self._get_all_trajectories()

        for trajectory in all_trajectories:
            visited = set()
            for _, cluster_id, _ in trajectory:
                visited.add(cluster_id)
            for cluster_id in visited:
                counts[cluster_id] += 1

        return dict(counts)

    def get_summary(self) -> dict:
        """
        Get a summary of all statistics.

        Returns:
            Dict with all computed statistics
        """
        return {
            "num_trajectories": len(self._get_all_trajectories()),
            "num_fork_points": len(self.fork_points),
            "num_clusters": self.metadata.get("num_clusters", 0),
            "convergence_rate": self.convergence_rate(),
            "average_trajectory_length": self.average_trajectory_length(),
            "graph_width": self.graph_width_over_time(),
            "entropy_over_time": self.trajectory_entropy_over_time(),
            "branching_factor": self.branching_factor(),
        }

    @staticmethod
    def independence_test_across_prompts(
        all_trajectories_list: list[dict],
    ) -> dict:
        """
        Test if trajectory distributions are independent across prompts.
        Uses chi-square test on cluster distributions.

        Args:
            all_trajectories_list: List of trajectory data dicts (one per prompt)

        Returns:
            {'statistic': float, 'p_value': float, 'independent': bool}
        """
        if len(all_trajectories_list) < 2:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "independent": True,
                "message": "Need at least 2 prompts for independence test",
            }

        # Collect cluster distributions per prompt
        all_clusters: set[int] = set()
        prompt_cluster_counts: list[dict[int, int]] = []

        for traj_data in all_trajectories_list:
            stats = TrajectoryStatistics(traj_data)
            counts = stats.cluster_visit_counts()
            prompt_cluster_counts.append(counts)
            all_clusters.update(counts.keys())

        # Build contingency table
        all_clusters = sorted(all_clusters)
        contingency = []
        for counts in prompt_cluster_counts:
            row = [counts.get(c, 0) for c in all_clusters]
            contingency.append(row)

        contingency = np.array(contingency)

        # Remove columns with all zeros
        non_zero_cols = contingency.sum(axis=0) > 0
        contingency = contingency[:, non_zero_cols]

        if contingency.shape[1] < 2:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "independent": True,
                "message": "Not enough clusters for chi-square test",
            }

        # Perform chi-square test
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)

        return {
            "statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "independent": p_value > 0.05,  # Standard significance level
        }


def get_convergence_info(
    streamlit_csv_file: str,
    base_data: dict,
    base_length_chars: int,
) -> dict:
    """
    Get convergence timestamp using utils/cot_analysis.py functions.

    Args:
        streamlit_csv_file: Path to the CSV file with outcome probabilities
        base_data: Base data dict for this prompt
        base_length_chars: Character length of base solution

    Returns:
        {
            'converged_timestep': int (in tokens),
            'converged_timestep_chars': int (approximate, in characters),
            'converged_timestep_norm': float,
            'converged_outcome': str,
            'is_correct': bool
        }
    """
    try:
        df = load_data("", file_template=streamlit_csv_file.replace("{idx}", ""))
    except Exception:
        # Try loading directly if template doesn't work
        import pandas as pd
        df = pd.read_csv(streamlit_csv_file)

    result = get_final_convergence(df)

    if result is None:
        return {
            "converged_timestep": None,
            "converged_timestep_chars": None,
            "converged_timestep_norm": None,
            "converged_outcome": None,
            "is_correct": None,
        }

    converged_t, converged_outcome = result

    # Convert token position to approximate character position
    # Use ratio of output_token_ids length to output_text length
    output_token_ids = base_data.get("output_token_ids", [])
    output_text = base_data.get("output_text", "")

    if output_token_ids and output_text:
        chars_per_token = len(output_text) / len(output_token_ids)
        converged_chars = int(converged_t * chars_per_token)
    else:
        converged_chars = converged_t  # Fallback

    converged_norm = converged_chars / base_length_chars if base_length_chars > 0 else 0.0

    # Check if correct
    correct_letter = base_data.get("correct_letter", "")
    is_correct = converged_outcome == correct_letter if correct_letter else None

    return {
        "converged_timestep": converged_t,
        "converged_timestep_chars": converged_chars,
        "converged_timestep_norm": converged_norm,
        "converged_outcome": converged_outcome,
        "is_correct": is_correct,
    }
