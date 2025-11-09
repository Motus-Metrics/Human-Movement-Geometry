# gait/step_detector.py
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple

def detect_steps_from_ankle_y(y: np.ndarray, fps: float, min_step_s: float = 0.45, max_step_s: float = 1.2) -> List[Tuple[int,int]]:
    """
    Detect gait cycles using minima of ankle vertical trajectory (side view).
    Returns list of (start_idx, end_idx) frames for each step.
    """
    y_inv = -y  # minima -> maxima
    peaks, _ = find_peaks(y_inv, distance=int(min_step_s*fps))
    # build steps between consecutive minima
    steps = []
    for i in range(1, len(peaks)):
        start, end = peaks[i-1], peaks[i]
        dur = (end-start)/fps
        if min_step_s <= dur <= max_step_s:
            steps.append((start, end))
    return steps

def resample_step(arr: np.ndarray, start: int, end: int, num: int = 101) -> np.ndarray:
    """
    arr: [T, D]; slice [start:end+1] -> resample to num points along time.
    """
    seg = arr[start:end+1]
    T = seg.shape[0]
    if T < 3:
        return np.repeat(seg[:1], num, axis=0)
    src = np.linspace(0, 1, T)
    tgt = np.linspace(0, 1, num)
    out = np.zeros((num, seg.shape[1]), dtype=float)
    for d in range(seg.shape[1]):
        out[:, d] = np.interp(tgt, src, seg[:, d])
    return out
