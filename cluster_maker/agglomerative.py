###
## cluster_maker - Agglomerative Clustering Algorithm
## Georgie Paterson - University of Bath
## December 2025
###


from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def _compute_centroids(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute pseudo-centroids as the mean of each cluster.
    This ensures compatibility with plotting_clustered.plot_clusters_2d(),
    which expects centroids like those returned by k-means.
    """
    unique_clusters = np.unique(labels)
    k = len(unique_clusters)
    n_features = X.shape[1]

    centroids = np.zeros((k, n_features))

    for i, cid in enumerate(unique_clusters):
        mask = labels == cid
        centroids[i] = X[mask].mean(axis=0)

    return centroids


def agglomerative(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
    distance_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical agglomerative clustering.

    Parameters
    ----------
    X : ndarray
        Feature matrix (already preprocessed by run_clustering).
    k : int
        Number of clusters to form (ignored if distance_threshold is set).
    linkage : str
        Linkage strategy: {"ward", "complete", "average", "single"}.
    distance_threshold : float or None

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    # Configure Agglomerative model
    model = AgglomerativeClustering(
        n_clusters=None if distance_threshold is not None else k,
        distance_threshold=distance_threshold,
        linkage=linkage,
    )

    labels = model.fit_predict(X)

    # Compute pseudo-centroids for pipeline compatibility
    centroids = _compute_centroids(X, labels)

    return labels, centroids
