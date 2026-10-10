"""Mining algorithms: K-Means/DBSCAN clustering, Isolation Forests, and PCA projection."""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def mine_operational_clusters(
    X: np.ndarray, n_clusters: int = 4
) -> Tuple[np.ndarray, KMeans]:
    """Partition operational modes (e.g. idle, ramp-up, full-load) via K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def detect_density_anomalies(
    X: np.ndarray, eps: float = 0.5, min_samples: int = 5
) -> np.ndarray:
    """Density-based spatial clustering to flag outlier sensor states (-1)."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels


def score_multivariate_anomalies(
    X: np.ndarray, contamination: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute continuous anomaly scores and binary flags using Isolation Forest."""
    iso_forest = IsolationForest(
        contamination=contamination, random_state=42, n_estimators=150
    )
    binary_flags = iso_forest.fit_predict(X)  # -1 is anomaly, 1 is normal
    scores = iso_forest.decision_function(X)  # Lower is more abnormal
    return binary_flags, scores


def reduce_dimensionality_pca(
    X: np.ndarray, n_components: int = 2
) -> Tuple[np.ndarray, PCA]:
    """Project high-dimensional sensor space for 2D/3D visualization."""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(X)
    return reduced, pca
