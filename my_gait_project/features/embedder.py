# features/embedder.py
from __future__ import annotations
import numpy as np
import umap

def umap_from_distance(
    D: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
):
    """
    Robustní UMAP pro předpočtené vzdálenosti i při malém počtu vzorků (N).
    Parametry se automaticky stáhnou tak, aby nepadal eigen-solver.
    """
    N = D.shape[0]
    if N < 3:
        raise ValueError(f"UMAP: potřebuju aspoň 3 body, mám {N}.")

    # n_components ≤ N-1
    n_components = min(n_components, max(1, N - 1))
    # n_neighbors ≤ N-1 a aspoň 2
    n_neighbors = min(n_neighbors, max(2, N - 1))

    # pro malé N je stabilnější random init (spektro by volalo eigsh s k>=N)
    init = "random" if N < 5 else "spectral"

    reducer = umap.UMAP(
        metric="precomputed",
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        init=init,
    )
    return reducer.fit_transform(D)
