# features/feature_maker.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from scipy.signal import savgol_filter

def make_step_features_xy(step_xy: np.ndarray, fps: float, use_z: bool = False) -> np.ndarray:
    """
    step_xy: [101, J*2]  or [101, J*3] if use_z=True
    Returns [101, D] where D = J*(2 or 3)*2  (pos + vel)
    """
    vel = savgol_filter(step_xy, 7, 2, deriv=1, delta=1.0/fps, axis=0, mode="interp")
    feat = np.concatenate([step_xy, vel], axis=1)
    return feat

def flatten_xyz(data: np.ndarray, axes: Tuple[int,...]=(0,1)) -> np.ndarray:
    """
    data: [T, J, D]; select axes (0:x,1:y,2:z), and flatten -> [T, J*len(axes)]
    """
    sel = data[:, :, list(axes)]
    T, J, A = sel.shape
    return sel.reshape(T, J*A)
