# pre/preprocessor.py
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter

def center_on_pelvis(data: np.ndarray, joint_names: list[str], pelvis_alias=("pelvis",)) -> np.ndarray:
    """
    data: [T,J,D]
    1) pokus o 'pelvis'
    2) fallback: průměr (left_hip, right_hip)
    3) pokud nic, vrátí data beze změny
    """
    nlow = [n.lower() for n in joint_names]

    def find_one(cands):
        for i, n in enumerate(nlow):
            if any(c in n for c in cands):
                return i
        return None

    pel_idx = find_one(["pelvis","hip_centre","hip_center","hips"])
    if pel_idx is not None:
        pelvis = data[:, pel_idx:pel_idx+1, :]
        return data - pelvis

    l_idx = find_one(["left_hip","lhip","hip_l"])
    r_idx = find_one(["right_hip","rhip","hip_r"])
    if l_idx is not None and r_idx is not None:
        mid = 0.5*(data[:, l_idx:l_idx+1, :] + data[:, r_idx:r_idx+1, :])
        return data - mid

    # fallback: necentruj
    return data

def scale_by_leg_length(data: np.ndarray, joint_names: list[str]) -> tuple[np.ndarray, float]:
    """
    Scale so that hip-ankle distance (median over time) ~ 1.0
    """
    def find(name_parts):
        for i, n in enumerate(joint_names):
            nlow = n.lower()
            if any(p in nlow for p in name_parts): return i
        return None

    hip = find(["right_hip","r_hip","hip_r","lefthip","l_hip","hip_l","hip"])
    ankle = find(["right_ankle","r_ankle","ankle_r","leftankle","l_ankle","ankle_l","ankle"])
    if hip is None or ankle is None:
        return data, 1.0
    dist = np.linalg.norm(data[:, hip, :] - data[:, ankle, :], axis=-1)
    s = np.median(dist[dist>0]) if np.any(dist>0) else 1.0
    return data / max(s, 1e-6), float(s)

def deriv_savgol(series: np.ndarray, window: int = 7, poly: int = 2, fps: float = 60.0) -> np.ndarray:
    """
    Derivative over time axis for features [T, D].
    Returns d/dt in same shape.
    """
    vel = savgol_filter(series, window_length=window, polyorder=poly, deriv=1, delta=1.0/fps, axis=0, mode="interp")
    return vel
