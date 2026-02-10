"""Abstract base class for circuit discovery algorithms."""

from abc import ABC, abstractmethod
from typing import List, Union

from transformers import PreTrainedModel, PreTrainedTokenizer

from utils.masks import EdgeMask, NodeMask
from utils.utils import Sentence


class CircuitDiscoveryAlgorithm(ABC):
    """Abstract base for circuit discovery methods.

    Subclasses implement `discover()` which returns a NodeMask (for within-layer
    attribution) or EdgeMask (for inter-layer attribution).
    """

    @abstractmethod
    def discover(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prefix_token_ids: List[int],
        branch_token_ids: List[List[int]],
        sentences: List[Sentence],
        layers: List[int],
        analysis_timestep: int,
        sentence_gap: int = 1,
        sentence_chunk: int = 1,
        objective_name: str = "kl_divergence",
        **kwargs,
    ) -> Union[NodeMask, EdgeMask]:
        """Run circuit discovery.

        Args:
            model: HF model with eager attention.
            tokenizer: Corresponding tokenizer.
            prefix_token_ids: Shared prefix tokens (prompt up to analysis point).
            branch_token_ids: List of N branch continuations (each is a list of token ids).
            sentences: Sentence boundaries over the prefix.
            layers: Which layers to analyze.
            analysis_timestep: Token position where analysis starts.
            sentence_gap: Edges between sentences < gap apart are always 1.
            sentence_chunk: Group consecutive sentences into chunks of this size.
            objective_name: Name of the loss function to use.

        Returns:
            NodeMask or EdgeMask with attribution scores.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm identifier string."""
        ...
