"""Data loader for forking paths data."""

import json
from collections import defaultdict
from typing import Any


class ForkingPathsLoader:
    """Load and preprocess forking paths data for a single prompt."""

    def __init__(self, forking_paths_file: str, base_data: dict):
        """
        Initialize the loader with forking paths file and base data.

        Args:
            forking_paths_file: Path to XX.json with rollouts
            base_data: Dict with 'output_text', 'output_token_ids' for this prompt
        """
        self.forking_paths_file = forking_paths_file
        self.base_data = base_data

        # Load forking paths data
        with open(forking_paths_file, "r") as f:
            self.rollouts = json.load(f)

        self._base_output_text = base_data["output_text"]
        self._base_length_chars = len(self._base_output_text)

    def get_base_output_text(self) -> str:
        """Return the base solution text."""
        return self._base_output_text

    def get_base_length_chars(self) -> int:
        """Return character length of base solution."""
        return self._base_length_chars

    def token_to_char_position(self, rollout: dict) -> int:
        """
        Convert token position to character position.

        Uses: char_pos = len(output_text) - len(post_stump_output_text)

        Args:
            rollout: A rollout dict containing 'output_text' and 'post_stump_output_text'

        Returns:
            Character position where the fork occurred
        """
        output_text = rollout["output_text"]
        post_stump_text = rollout["post_stump_output_text"]
        return len(output_text) - len(post_stump_text)

    def get_rollouts_by_fork_point(self) -> dict[int, list[dict]]:
        """
        Group rollouts by their fork character position.

        Returns:
            {char_pos: [{'stump_text': str, 'post_stump_text': str,
                         'full_text': str, 'answer': str, 't_token': int}, ...]}
        """
        grouped: dict[int, list[dict]] = defaultdict(list)

        for rollout in self.rollouts:
            char_pos = self.token_to_char_position(rollout)
            stump_text = rollout["output_text"][: char_pos]
            post_stump_text = rollout["post_stump_output_text"]

            grouped[char_pos].append({
                "stump_text": stump_text,
                "post_stump_text": post_stump_text,
                "full_text": rollout["output_text"],
                "answer": rollout.get("clean_answer", ""),
                "t_token": rollout["t"],
            })

        return dict(grouped)

    def get_all_rollouts(self) -> list[dict]:
        """Return all rollouts as a flat list."""
        return self.rollouts

    def get_num_rollouts(self) -> int:
        """Return total number of rollouts."""
        return len(self.rollouts)

    def get_unique_fork_points(self) -> list[int]:
        """Return sorted list of unique fork character positions."""
        fork_points = set()
        for rollout in self.rollouts:
            char_pos = self.token_to_char_position(rollout)
            fork_points.add(char_pos)
        return sorted(fork_points)


def load_base_data(base_data_file: str) -> list[dict]:
    """
    Load base data from a JSON file.

    Args:
        base_data_file: Path to base_data.json

    Returns:
        List of base data dicts, one per prompt
    """
    with open(base_data_file, "r") as f:
        return json.load(f)


def get_forking_paths_files(forking_paths_dir: str) -> list[str]:
    """
    Get all forking paths JSON files in a directory.

    Args:
        forking_paths_dir: Directory containing XX.json files

    Returns:
        Sorted list of file paths
    """
    import os

    files = []
    for filename in os.listdir(forking_paths_dir):
        if filename.endswith(".json"):
            files.append(os.path.join(forking_paths_dir, filename))
    return sorted(files)
