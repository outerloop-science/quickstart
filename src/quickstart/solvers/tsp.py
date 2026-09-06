"""TSP solver. Baseline: nearest neighbor from city 0.

Improvement ladder (suggestions, not limits): 2-opt, or-opt, better starts,
Lin-Kernighan-style moves. `solve` must return a permutation of range(n).
"""

from __future__ import annotations

import numpy as np


def solve(coords: np.ndarray) -> list[int]:
    """Return a tour visiting every city exactly once."""
    n = coords.shape[0]
    unvisited = set(range(1, n))
    tour = [0]
    while unvisited:
        cur = coords[tour[-1]]
        nxt = min(unvisited, key=lambda j: float(np.sum((coords[j] - cur) ** 2)))
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour
