from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine, euclidean


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    d = euclidean(_z(a), _z(b))
    return float(1.0 / (1.0 + d / max(len(a), 1) ** 0.5))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    value = 1.0 - cosine(_z(a), _z(b))
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


def correlation_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = _z(a), _z(b)
    value = np.corrcoef(aa, bb)[0, 1] if len(aa) > 1 else 0.0
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compact multivariate DTW implementation for research-scale candidates."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    dp = np.full((len(x) + 1, len(y) + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            cost = float(np.linalg.norm(x[i - 1] - y[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[-1, -1])


def ensemble_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Structural score only; it is deliberately not a profitability probability."""
    e = euclidean_similarity(a, b)
    c = cosine_similarity(a, b)
    r = correlation_similarity(a, b)
    d = dtw_distance(a.reshape(-1, 1), b.reshape(-1, 1))
    dtw_score = 1.0 / (1.0 + d / max(len(a), len(b), 1) ** 0.5)
    return float(np.clip(0.30 * e + 0.25 * c + 0.20 * r + 0.25 * dtw_score, 0.0, 1.0))
