from __future__ import annotations

import numpy as np

from patterns.similarity import ensemble_similarity


class HistoricalSearcher:
    def __init__(self, minimum_similarity: float = 0.85, top_k: int = 10) -> None:
        self.minimum_similarity = minimum_similarity
        self.top_k = top_k
        self._items: list[tuple[int, int, np.ndarray, dict]] = []

    def add(self, event_id: int, lookback: int, vector: np.ndarray, metadata: dict | None = None) -> None:
        self._items.append((event_id, lookback, np.asarray(vector, dtype=float), metadata or {}))

    def search(self, vector: np.ndarray, lookback: int | None = None) -> list[dict]:
        query = np.asarray(vector, dtype=float)
        scored: list[dict] = []
        for event_id, lb, candidate, metadata in self._items:
            if lookback is not None and lb != lookback:
                continue
            if len(candidate) != len(query):
                continue
            score = ensemble_similarity(query, candidate)
            if score >= self.minimum_similarity:
                scored.append({"event_id": event_id, "lookback": lb, "similarity": score, **metadata})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[: self.top_k]
