# features/spd_geom.py
from __future__ import annotations
import numpy as np
from typing import List, Sequence
from geomstats.geometry.spd_matrices import SPDMatrices

# Rozpoznání API geomstats (metrika)
try:
    # geomstats >= 2.8: metrika bere SPACE
    from geomstats.geometry.spd_matrices import SPDAffineMetric  # type: ignore
    _NEW_API = True
except ImportError:
    # starší geomstats: metrika bere DIMENZI
    from geomstats.geometry.spd_matrices import SPDMetricAffine  # type: ignore
    _NEW_API = False

EPS = 1e-6


class SPDGeom:
    """SPD prostor + metriky, kompatibilní napříč verzemi geomstats."""

    def __init__(self, dim: int):
        self.M = SPDMatrices(dim)
        if _NEW_API:
            # novější geomstats: metrika dostává prostor
            self.metric = SPDAffineMetric(self.M)  # type: ignore[arg-type]
        else:
            # starší geomstats: metrika dostává dimenzi
            self.metric = SPDMetricAffine(dim)  # type: ignore[call-arg]

    def dist(self, A: np.ndarray, B: np.ndarray) -> float:
        return float(self.metric.dist(A, B))

    def mean(
        self,
        mats: Sequence[np.ndarray],
        max_iter: int = 64,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """
        Fréchetův (Karcherův) průměr na SPD přes vlastní iterátor.
        Stabilní napříč verzemi geomstats (bez FrechetMean).
        """
        if len(mats) == 1:
            return np.array(mats[0], dtype=float)

        # inicializace: první matice
        mu = np.array(mats[0], dtype=float)

        for _ in range(max_iter):
            # průměr log-map v tečném prostoru v mu
            logs = [self.metric.log(C, mu) for C in mats]
            avg = np.mean(logs, axis=0)

            step_norm = float(np.linalg.norm(avg))
            if step_norm < tol:
                break

            # krok po geodetice
            mu = self.metric.exp(avg, mu)

        return mu


def spd_from_features(feat: np.ndarray) -> np.ndarray:
    """
    feat: [T, d]  -> kovariance přes fázi (SPD matice [d,d]).
    """
    C = np.cov(feat.T)
    d = C.shape[0]
    return C + EPS * np.eye(d)


def spd_sequence(feat: np.ndarray, win: int = 11) -> List[np.ndarray]:
    """
    feat: [T, d] -> list SPD matic z klouzavého okna po fázi.
    """
    T, d = feat.shape
    half = max(1, win // 2)
    seq: List[np.ndarray] = []
    for t in range(T):
        a = max(0, t - half)
        b = min(T, t + half + 1)
        C = np.cov(feat[a:b].T)
        seq.append(C + EPS * np.eye(d))
    return seq


def smooth_length(seq: List[np.ndarray], geom: SPDGeom) -> float:
    """
    Riemannovská délka SPD trajektorie ~ suma geodetických kroků.
    """
    if len(seq) < 2:
        return 0.0
    return float(np.sum([geom.dist(seq[i], seq[i - 1]) for i in range(1, len(seq))]))


def avg_step_velocity(seq: List[np.ndarray], geom: SPDGeom) -> float:
    """
    Průměrná 'rychlost změn' na varietě (na vzorek fáze).
    """
    if len(seq) < 2:
        return 0.0
    return smooth_length(seq, geom) / (len(seq) - 1)


def frechet_variance(mats: List[np.ndarray], geom: SPDGeom) -> float:
    """
    Fréchetova variance napříč kroky (stabilita/variabilita).
    """
    if len(mats) == 0:
        return 0.0
    mu = geom.mean(mats)
    return float(np.mean([geom.dist(C, mu) ** 2 for C in mats]))


def pairwise_dist(mats: List[np.ndarray], geom: SPDGeom) -> np.ndarray:
    """
    Předpočtená matice SPD vzdáleností (pro UMAP metric='precomputed').
    """
    n = len(mats)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = geom.dist(mats[i], mats[j])
            D[i, j] = D[j, i] = d
    return D
