"""Embedding models for semantic sentence embeddings."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models (extensible)."""

    @abstractmethod
    def embed(self, sentences: list[str]) -> np.ndarray:
        """
        Embed a list of sentences.

        Args:
            sentences: List of sentences to embed

        Returns:
            (N, dim) numpy array of embeddings
        """
        pass

    @abstractmethod
    def embed_single(self, sentence: str) -> np.ndarray:
        """
        Embed a single sentence.

        Args:
            sentence: Sentence to embed

        Returns:
            (dim,) numpy array embedding
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Wrapper around sentence-transformers library."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Load the sentence transformer model.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cpu', 'cuda', etc.). None for auto-detect.
            batch_size: Batch size for encoding
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(self, sentences: list[str]) -> np.ndarray:
        """
        Batch embed sentences.

        Args:
            sentences: List of sentences to embed

        Returns:
            (N, dim) numpy array of embeddings
        """
        if not sentences:
            return np.array([]).reshape(0, self._embedding_dim)

        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=len(sentences) > 100,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_single(self, sentence: str) -> np.ndarray:
        """
        Embed single sentence.

        Args:
            sentence: Sentence to embed

        Returns:
            (dim,) numpy array embedding
        """
        embedding = self.model.encode(
            sentence,
            convert_to_numpy=True,
        )
        return embedding

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim
