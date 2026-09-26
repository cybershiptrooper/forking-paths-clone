"""Clustering engines for semantic clustering."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from tqdm import tqdm


class BaseClusteringEngine(ABC):
    """Abstract base class for clustering algorithms (extensible)."""

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit clustering model to embeddings.

        Args:
            embeddings: (N, dim) array of embeddings
        """
        pass

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster IDs for embeddings.

        Args:
            embeddings: (N, dim) array of embeddings

        Returns:
            (N,) array of cluster IDs
        """
        pass

    @abstractmethod
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.

        Args:
            embeddings: (N, dim) array of embeddings

        Returns:
            (N,) array of cluster IDs
        """
        pass

    @abstractmethod
    def get_num_clusters(self) -> int:
        """Return number of clusters found."""
        pass


class AgglomerativeClusteringEngine(BaseClusteringEngine):
    """Agglomerative clustering using sklearn."""

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        distance_threshold: float = 1.0,
        linkage: str = "average",
        metric: str = "cosine",
    ):
        """
        Initialize agglomerative clustering.

        Args:
            n_clusters: Number of clusters (None to use distance_threshold)
            distance_threshold: Distance threshold for merging (used if n_clusters=None)
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
                     Note: 'ward' only works with euclidean metric
            metric: Distance metric ('cosine', 'euclidean', etc.)
        """
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.metric = metric

        # Computed after fit
        self._labels: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None
        self._fitted_embeddings: Optional[np.ndarray] = None
        self._num_clusters: int = 0

    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit agglomerative clustering.

        Args:
            embeddings: (N, dim) array of embeddings
        """
        if embeddings.shape[0] == 0:
            self._labels = np.array([], dtype=int)
            self._centroids = np.array([]).reshape(0, embeddings.shape[1] if len(embeddings.shape) > 1 else 0)
            self._num_clusters = 0
            return

        # Create clustering model
        # Ward linkage requires euclidean metric
        if self.linkage == "ward":
            clustering = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                distance_threshold=self.distance_threshold if self.n_clusters is None else None,
                linkage=self.linkage,
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                distance_threshold=self.distance_threshold if self.n_clusters is None else None,
                linkage=self.linkage,
                metric=self.metric,
            )

        self._labels = clustering.fit_predict(embeddings)
        self._fitted_embeddings = embeddings
        self._num_clusters = len(np.unique(self._labels))

        # Compute centroids
        self._compute_centroids(embeddings)

    def _compute_centroids(self, embeddings: np.ndarray) -> None:
        """Compute cluster centroids."""
        if self._labels is None or len(self._labels) == 0:
            return

        unique_labels = np.unique(self._labels)
        centroids = []
        for label in tqdm(unique_labels, desc="Computing centroids"):
            mask = self._labels == label
            cluster_embeddings = embeddings[mask]
            centroid = cluster_embeddings.mean(axis=0)
            centroids.append(centroid)

        self._centroids = np.array(centroids)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict using nearest centroid.

        Args:
            embeddings: (N, dim) array of embeddings

        Returns:
            (N,) array of cluster IDs
        """
        if self._centroids is None or len(self._centroids) == 0:
            raise ValueError("Model not fitted. Call fit() first.")

        if embeddings.shape[0] == 0:
            return np.array([], dtype=int)

        # Compute distances to centroids
        if self.metric == "cosine":
            distances = cosine_distances(embeddings, self._centroids)
        else:
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(embeddings, self._centroids)

        # Assign to nearest centroid
        return np.argmin(distances, axis=1)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit and return cluster labels.

        Args:
            embeddings: (N, dim) array of embeddings

        Returns:
            (N,) array of cluster IDs
        """
        self.fit(embeddings)
        return self._labels if self._labels is not None else np.array([], dtype=int)

    def get_num_clusters(self) -> int:
        """Return number of clusters."""
        return self._num_clusters

    def get_cluster_centroids(self) -> np.ndarray:
        """
        Return cluster centroids.

        Returns:
            (num_clusters, dim) array of centroids
        """
        if self._centroids is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._centroids

    def get_labels(self) -> np.ndarray:
        """
        Return fitted cluster labels.

        Returns:
            (N,) array of cluster IDs
        """
        if self._labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._labels

    def get_representative_sentences(
        self, sentences: list[str], embeddings: np.ndarray
    ) -> dict[int, str]:
        """
        Return most representative sentence per cluster (closest to centroid).

        Args:
            sentences: List of sentences (same order as embeddings)
            embeddings: (N, dim) array of embeddings

        Returns:
            {cluster_id: representative_sentence}
        """
        if self._centroids is None or self._labels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        representatives = {}
        unique_labels = np.unique(self._labels)

        for label in unique_labels:
            mask = self._labels == label
            cluster_embeddings = embeddings[mask]
            cluster_sentences = [s for s, m in zip(sentences, mask) if m]

            # Find sentence closest to centroid
            centroid = self._centroids[label]
            if self.metric == "cosine":
                distances = cosine_distances(cluster_embeddings, centroid.reshape(1, -1)).flatten()
            else:
                from sklearn.metrics.pairwise import euclidean_distances
                distances = euclidean_distances(cluster_embeddings, centroid.reshape(1, -1)).flatten()

            closest_idx = np.argmin(distances)
            representatives[int(label)] = cluster_sentences[closest_idx]

        return representatives

    def get_cluster_sentences(
        self, sentences: list[str]
    ) -> dict[int, list[str]]:
        """
        Return all sentences per cluster.

        Args:
            sentences: List of sentences (same order as fitted embeddings)

        Returns:
            {cluster_id: [sentences in cluster]}
        """
        if self._labels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        cluster_sentences: dict[int, list[str]] = {}
        for sentence, label in zip(sentences, self._labels):
            label = int(label)
            if label not in cluster_sentences:
                cluster_sentences[label] = []
            cluster_sentences[label].append(sentence)

        return cluster_sentences


class MiniBatchKMeansClusteringEngine(BaseClusteringEngine):
    """Fast clustering using MiniBatchKMeans - suitable for large datasets (100k+ sentences)."""

    def __init__(
        self,
        n_clusters: int = 100,
        batch_size: int = 1024,
        random_state: int = 42,
        max_iter: int = 100,
    ):
        """
        Initialize MiniBatchKMeans clustering.

        Args:
            n_clusters: Number of clusters to create
            batch_size: Size of mini batches for training
            random_state: Random seed for reproducibility
            max_iter: Maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_iter = max_iter

        # Computed after fit
        self._model: Optional[MiniBatchKMeans] = None
        self._labels: Optional[np.ndarray] = None
        self._num_clusters: int = 0

    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit MiniBatchKMeans clustering.

        Args:
            embeddings: (N, dim) array of embeddings
        """
        if embeddings.shape[0] == 0:
            self._labels = np.array([], dtype=int)
            self._num_clusters = 0
            return

        # Adjust n_clusters if we have fewer samples
        actual_n_clusters = min(self.n_clusters, embeddings.shape[0])

        self._model = MiniBatchKMeans(
            n_clusters=actual_n_clusters,
            batch_size=min(self.batch_size, embeddings.shape[0]),
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init="auto",
        )
        self._labels = self._model.fit_predict(embeddings)
        self._num_clusters = actual_n_clusters

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster IDs for new embeddings.

        Args:
            embeddings: (N, dim) array of embeddings

        Returns:
            (N,) array of cluster IDs
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if embeddings.shape[0] == 0:
            return np.array([], dtype=int)

        return self._model.predict(embeddings)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit and return cluster labels.

        Args:
            embeddings: (N, dim) array of embeddings

        Returns:
            (N,) array of cluster IDs
        """
        self.fit(embeddings)
        return self._labels if self._labels is not None else np.array([], dtype=int)

    def get_num_clusters(self) -> int:
        """Return number of clusters."""
        return self._num_clusters

    def get_cluster_centroids(self) -> np.ndarray:
        """
        Return cluster centroids.

        Returns:
            (num_clusters, dim) array of centroids
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._model.cluster_centers_

    def get_labels(self) -> np.ndarray:
        """
        Return fitted cluster labels.

        Returns:
            (N,) array of cluster IDs
        """
        if self._labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._labels

    def get_representative_sentences(
        self, sentences: list[str], embeddings: np.ndarray
    ) -> dict[int, str]:
        """
        Return most representative sentence per cluster (closest to centroid).

        Args:
            sentences: List of sentences (same order as embeddings)
            embeddings: (N, dim) array of embeddings

        Returns:
            {cluster_id: representative_sentence}
        """
        if self._model is None or self._labels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        representatives = {}
        centroids = self._model.cluster_centers_

        for label in np.unique(self._labels):
            mask = self._labels == label
            cluster_embeddings = embeddings[mask]
            cluster_sentences = [s for s, m in zip(sentences, mask) if m]

            # Find sentence closest to centroid
            distances = np.linalg.norm(cluster_embeddings - centroids[label], axis=1)
            closest_idx = np.argmin(distances)
            representatives[int(label)] = cluster_sentences[closest_idx]

        return representatives

    def get_cluster_sentences(self, sentences: list[str]) -> dict[int, list[str]]:
        """
        Return all sentences per cluster.

        Args:
            sentences: List of sentences (same order as fitted embeddings)

        Returns:
            {cluster_id: [sentences in cluster]}
        """
        if self._labels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        cluster_sentences: dict[int, list[str]] = {}
        for sentence, label in zip(sentences, self._labels):
            label = int(label)
            if label not in cluster_sentences:
                cluster_sentences[label] = []
            cluster_sentences[label].append(sentence)

        return cluster_sentences
