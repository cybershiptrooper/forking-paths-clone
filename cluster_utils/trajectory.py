"""Trajectory building from forking paths data."""

import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Import from parent directory
sys.path.insert(0, "..")
from utils.cot_analysis import split_into_sentences

from .clustering import BaseClusteringEngine
from .data_loader import ForkingPathsLoader
from .embedding import BaseEmbeddingModel


@dataclass
class SentenceInfo:
    """Information about a sentence in a trajectory."""

    text: str
    start_char: int
    end_char: int
    cluster_id: int = -1  # -1 means not yet assigned


@dataclass
class Trajectory:
    """A single trajectory (base or rollout)."""

    fork_char_pos: int  # Character position where this trajectory forked
    stump_sentence: SentenceInfo  # The sentence before the fork (shared starting point)
    sentences: list[SentenceInfo] = field(default_factory=list)  # All sentences after fork
    answer: str = ""  # Final answer (clean_answer)

    def to_normalized_list(
        self, base_length: int
    ) -> list[tuple[float, int, str]]:
        """
        Convert to [(t_norm, cluster_id, sentence_text), ...] format.
        Merges consecutive sentences with same cluster_id.

        Args:
            base_length: Character length of base solution for normalization

        Returns:
            List of (t_norm, cluster_id, text) tuples with consecutive same-cluster merged
        """
        if not self.sentences:
            # Only have stump sentence
            t_norm = self.stump_sentence.start_char / base_length
            return [(t_norm, self.stump_sentence.cluster_id, self.stump_sentence.text)]

        result = []

        # Start with stump sentence
        current_cluster = self.stump_sentence.cluster_id
        current_text = self.stump_sentence.text
        current_start = self.stump_sentence.start_char

        # Process all sentences
        for sent in self.sentences:
            if sent.cluster_id == current_cluster:
                # Same cluster - merge text
                current_text += " " + sent.text
            else:
                # Different cluster - save current and start new
                t_norm = current_start / base_length
                result.append((t_norm, current_cluster, current_text))
                current_cluster = sent.cluster_id
                current_text = sent.text
                current_start = sent.start_char

        # Add final merged entry
        t_norm = current_start / base_length
        result.append((t_norm, current_cluster, current_text))

        return result


class TrajectoryBuilder:
    """Build trajectories from forking paths data."""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        clustering_engine: BaseClusteringEngine,
    ):
        """
        Initialize with embedding and clustering components.

        Args:
            embedding_model: Model for embedding sentences
            clustering_engine: Engine for clustering embeddings
        """
        self.embedding_model = embedding_model
        self.clustering_engine = clustering_engine

        # Cached data after building
        self._all_sentences: list[str] = []
        self._all_embeddings: Optional[np.ndarray] = None
        self._sentence_to_cluster: dict[str, int] = {}

    def build_for_prompt(self, loader: ForkingPathsLoader) -> dict:
        """
        Build all trajectories for a single prompt.

        Args:
            loader: ForkingPathsLoader with the data

        Returns:
            {
                "metadata": {...},
                "cluster_sentences": {cluster_id: [sentences]},
                "representative_sentences": {cluster_id: sentence},
                "fork_points": {
                    fork_char_pos: {
                        "base_sentence": (sentence_text, cluster_id),
                        "trajectory_0": [(t_norm, cluster_id, text), ...],
                        ...
                    }
                }
            }
        """
        base_text = loader.get_base_output_text()
        base_length = loader.get_base_length_chars()
        rollouts_by_fork = loader.get_rollouts_by_fork_point()

        # Step 1: Collect all unique sentences from base and all rollouts
        all_sentences = self._collect_all_sentences(base_text, rollouts_by_fork)

        # Step 2: Embed all sentences
        unique_sentences = list(set(all_sentences))
        embeddings = self.embedding_model.embed(unique_sentences)

        # Step 3: Cluster embeddings
        labels = self.clustering_engine.fit_predict(embeddings)

        # Build sentence -> cluster mapping
        self._sentence_to_cluster = {
            sent: int(label) for sent, label in zip(unique_sentences, labels)
        }

        # Step 4: Build trajectories for each fork point
        fork_points = {}

        # Get base sentences with positions
        base_sentences_with_pos = self._split_and_locate_sentences(base_text)

        for fork_char_pos, rollouts in rollouts_by_fork.items():
            fork_data = {}

            # Find the stump sentence (last complete sentence before fork)
            stump_sentence = self._find_stump_sentence(
                base_sentences_with_pos, fork_char_pos
            )

            # Store base sentence info
            fork_data["base_sentence"] = (
                stump_sentence.text,
                self._sentence_to_cluster.get(stump_sentence.text, -1),
            )

            # Build trajectory for each rollout
            for i, rollout in enumerate(rollouts):
                trajectory = self._build_single_trajectory(
                    rollout, fork_char_pos, stump_sentence, base_length
                )
                fork_data[f"trajectory_{i}"] = trajectory.to_normalized_list(base_length)

            fork_points[str(fork_char_pos)] = fork_data

        # Get cluster info
        cluster_sentences = self.clustering_engine.get_cluster_sentences(unique_sentences)
        representative_sentences = self.clustering_engine.get_representative_sentences(
            unique_sentences, embeddings
        )

        return {
            "metadata": {
                "base_length_chars": base_length,
                "num_clusters": self.clustering_engine.get_num_clusters(),
                "embedding_model": getattr(
                    self.embedding_model, "model_name", "unknown"
                ),
            },
            "cluster_sentences": {
                str(k): v for k, v in cluster_sentences.items()
            },
            "representative_sentences": {
                str(k): v for k, v in representative_sentences.items()
            },
            "fork_points": fork_points,
        }

    def _collect_all_sentences(
        self, base_text: str, rollouts_by_fork: dict[int, list[dict]]
    ) -> list[str]:
        """
        Collect all sentences from base text and all rollouts.

        Args:
            base_text: Base solution text
            rollouts_by_fork: Rollouts grouped by fork point

        Returns:
            List of all sentences (may contain duplicates)
        """
        all_sentences = []

        # Base sentences
        base_sentences = split_into_sentences(base_text)
        all_sentences.extend(base_sentences)

        # Rollout sentences
        for fork_pos, rollouts in rollouts_by_fork.items():
            for rollout in rollouts:
                full_text = rollout["full_text"]
                sentences = split_into_sentences(full_text)
                all_sentences.extend(sentences)

        return all_sentences

    def _split_and_locate_sentences(
        self, text: str
    ) -> list[tuple[str, int, int]]:
        """
        Split text into sentences with character positions.

        Uses split_into_sentences() from utils/cot_analysis.py and
        finds positions by searching in original text.

        Args:
            text: Text to split

        Returns:
            [(sentence, start_char, end_char), ...]
        """
        sentences = split_into_sentences(text)
        result = []

        search_start = 0
        for sentence in sentences:
            # Find sentence in text
            start_idx = text.find(sentence, search_start)
            if start_idx == -1:
                # Try finding with stripped whitespace variations
                start_idx = search_start
            end_idx = start_idx + len(sentence)
            result.append((sentence, start_idx, end_idx))
            search_start = end_idx

        return result

    def _find_stump_sentence(
        self,
        sentences_with_pos: list[tuple[str, int, int]],
        fork_char_pos: int,
    ) -> SentenceInfo:
        """
        Find the last complete sentence before fork_char_pos.

        Args:
            sentences_with_pos: List of (sentence, start, end) tuples
            fork_char_pos: Character position of the fork

        Returns:
            SentenceInfo for the stump sentence
        """
        # Find the last sentence that ends at or before fork_char_pos
        stump_sent = None
        for sentence, start, end in sentences_with_pos:
            if end <= fork_char_pos:
                stump_sent = SentenceInfo(
                    text=sentence,
                    start_char=start,
                    end_char=end,
                    cluster_id=self._sentence_to_cluster.get(sentence, -1),
                )
            else:
                break

        if stump_sent is None:
            # Use first sentence if no sentence ends before fork
            if sentences_with_pos:
                sentence, start, end = sentences_with_pos[0]
                stump_sent = SentenceInfo(
                    text=sentence,
                    start_char=start,
                    end_char=end,
                    cluster_id=self._sentence_to_cluster.get(sentence, -1),
                )
            else:
                stump_sent = SentenceInfo(
                    text="",
                    start_char=0,
                    end_char=0,
                    cluster_id=-1,
                )

        return stump_sent

    def _build_single_trajectory(
        self,
        rollout: dict,
        fork_char_pos: int,
        stump_sentence: SentenceInfo,
        base_length: int,
    ) -> Trajectory:
        """
        Build a Trajectory object for a single rollout.

        Args:
            rollout: Rollout dict with full_text, answer, etc.
            fork_char_pos: Character position where fork occurred
            stump_sentence: The shared starting sentence
            base_length: Base solution length for normalization

        Returns:
            Trajectory object
        """
        full_text = rollout["full_text"]
        answer = rollout.get("answer", "")

        # Get all sentences with positions
        sentences_with_pos = self._split_and_locate_sentences(full_text)

        # Build sentence infos for sentences after the fork
        sentences = []
        for sentence, start, end in sentences_with_pos:
            # Only include sentences that start at or after the fork position
            if start >= fork_char_pos:
                sent_info = SentenceInfo(
                    text=sentence,
                    start_char=start,
                    end_char=end,
                    cluster_id=self._sentence_to_cluster.get(sentence, -1),
                )
                sentences.append(sent_info)

        # Create trajectory with updated stump sentence cluster
        stump_with_cluster = SentenceInfo(
            text=stump_sentence.text,
            start_char=stump_sentence.start_char,
            end_char=stump_sentence.end_char,
            cluster_id=self._sentence_to_cluster.get(stump_sentence.text, -1),
        )

        return Trajectory(
            fork_char_pos=fork_char_pos,
            stump_sentence=stump_with_cluster,
            sentences=sentences,
            answer=answer,
        )

    def get_sentence_to_cluster_map(self) -> dict[str, int]:
        """Return the sentence to cluster mapping."""
        return self._sentence_to_cluster
