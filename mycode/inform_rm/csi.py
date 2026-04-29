"""Cluster Separation Index (CSI) utilities for InfoRM latent-space monitoring."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
except ImportError:  # pragma: no cover - optional dependency
    SklearnDBSCAN = None



def _ensure_2d(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("Point array is empty")
    return arr


def _dbscan_numpy(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Small fallback DBSCAN implementation for environments without scikit-learn."""

    unassigned = -2
    noise = -1
    labels = np.full(points.shape[0], unassigned, dtype=np.int32)

    def region_query(idx: int) -> np.ndarray:
        distances = np.linalg.norm(points - points[idx], axis=1)
        return np.where(distances <= eps)[0]

    cluster_id = 0
    for idx in range(points.shape[0]):
        if labels[idx] != unassigned:
            continue
        neighbors = region_query(idx)
        if neighbors.size < min_samples:
            labels[idx] = noise
            continue

        labels[idx] = cluster_id
        seeds = [int(item) for item in neighbors.tolist() if int(item) != idx]
        seen = set(seeds)
        cursor = 0
        while cursor < len(seeds):
            point_idx = seeds[cursor]
            cursor += 1
            if labels[point_idx] == noise:
                labels[point_idx] = cluster_id
            if labels[point_idx] != unassigned:
                continue
            labels[point_idx] = cluster_id
            point_neighbors = region_query(point_idx)
            if point_neighbors.size >= min_samples:
                for neighbor in point_neighbors.tolist():
                    neighbor = int(neighbor)
                    if neighbor not in seen and neighbor != idx:
                        seeds.append(neighbor)
                        seen.add(neighbor)
        cluster_id += 1
    return labels


def _dbscan_labels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if SklearnDBSCAN is not None:
        return SklearnDBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    return _dbscan_numpy(points, eps=eps, min_samples=min_samples)



def cluster_separation_index(
    red_points: np.ndarray,
    blue_points: np.ndarray,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    ignore_noise: bool = False,
) -> Tuple[float, int]:
    red = _ensure_2d(red_points)
    blue = _ensure_2d(blue_points)
    labels = _dbscan_labels(red, eps=dbscan_eps, min_samples=dbscan_min_samples)

    total = 0.0
    cluster_count = 0
    for cluster_id in sorted(set(labels.tolist())):
        if ignore_noise and cluster_id == -1:
            continue
        cluster_points = red[labels == cluster_id]
        if cluster_points.size == 0:
            continue
        center = cluster_points.mean(axis=0)
        distances = np.linalg.norm(blue - center[None, :], axis=1)
        total += float(np.min(distances) * len(cluster_points))
        cluster_count += 1
    return total, cluster_count



def compute_csi(
    red_points: Iterable[Iterable[float]],
    blue_points: Iterable[Iterable[float]],
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    ignore_noise: bool = False,
) -> float:
    value, _ = cluster_separation_index(
        np.asarray(list(red_points), dtype=np.float32),
        np.asarray(list(blue_points), dtype=np.float32),
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        ignore_noise=ignore_noise,
    )
    return value
