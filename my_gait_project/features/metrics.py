# features/metrics.py
import numpy as np
from __future__ import annotations
from typing import List
import numpy as np
from .spd_geom import SPDGeom, smooth_length, avg_step_velocity, frechet_variance, symmetry_index, pairwise_dist

class Metrics:
    def traj_length(self, traj_2d: np.ndarray) -> float:
        dif = np.diff(traj_2d, axis=0)
        return float(np.sum(np.linalg.norm(dif, axis=1)))

    def mean_curvature(self, traj_2d: np.ndarray) -> float:
        v = np.gradient(traj_2d, axis=0)
        a = np.gradient(v, axis=0)
        num = np.abs(v[:,0]*a[:,1] - v[:,1]*a[:,0])
        den = (np.linalg.norm(v, axis=1)**3 + 1e-8)
        kappa = num / den
        return float(np.nanmean(kappa))

    def variability_sd(self, coords_2d: np.ndarray) -> float:
        mu = np.nanmean(coords_2d, axis=0, keepdims=True)
        d = np.linalg.norm(coords_2d - mu, axis=1)
        return float(np.nanstd(d))
