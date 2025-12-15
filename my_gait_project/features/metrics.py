Copyright (c) 2025 Thomas Boozek
SPDX-License-Identifier: AGPL-3.0-only

# features/metrics.py
from __future__ import annotations
from typing import List
import numpy as np
from .spd_geom import SPDGeom, smooth_length, avg_step_velocity, frechet_variance, pairwise_dist

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

def calculate_euclidean_metrics(step_data_matrix):
    """
    Vypočítá Euklidovské metriky pro jeden krok (step_data_matrix).

    Args:
        step_data_matrix (np.array): Matice tvaru (n_frames, n_features).
                                     Obsahuje souřadnice [x, y] kloubů v čase.

    Returns:
        tuple: (euclid_smooth, euclid_v_bar, euclid_var)
    """
    # 1. Rychlosti (změna polohy mezi snímky)
    # axis=0 znamená rozdíl mezi řádky (časem)
    velocities = np.diff(step_data_matrix, axis=0)

    # Velikost rychlosti v každém časovém okamžiku (Euklidovská norma vektoru rychlosti všech kloubů)
    speed_norms = np.linalg.norm(velocities, axis=1)

    # --- METRIKA 1: Smoothness (Celková délka trajektorie v Euklidovském prostoru) ---
    # Součet všech pohybů
    e_smooth = np.sum(speed_norms)

    # --- METRIKA 2: Mean Velocity (Průměrná rychlost) ---
    # Průměr velikostí rychlostí
    e_v_bar = np.mean(speed_norms)

    # --- METRIKA 3: Variance (Celkový rozptyl dat v prostoru) ---
    # Total Variation = Stopa kovarianční matice (součet rozptylů jednotlivých souřadnic)
    # Měří, jak moc se "mračno bodů" (tvar kroku) rozprostírá kolem svého průměru
    centered = step_data_matrix - np.mean(step_data_matrix, axis=0)
    # rowvar=False, protože řádky jsou pozorování (čas) a sloupce jsou proměnné (souřadnice)
    cov_matrix = np.cov(centered, rowvar=False)
    e_var = np.trace(cov_matrix)

    return e_smooth, e_v_bar, e_var
