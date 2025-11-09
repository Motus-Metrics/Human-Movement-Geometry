from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

FACTORIAL_JOINT_ALIASES = {
    # pravá DK
    "pelvis": ["pelvis", "Pelvis", "hip_center", "HipCentre", "Hips"],
    "r_hip":  ["right_hip", "RightHip", "r_hip", "RHip", "hip_r"],
    "r_knee": ["right_knee", "RightKnee", "r_knee", "RKnee", "knee_r"],
    "r_ankle":["right_ankle","RightAnkle","r_ankle","RAnkle","ankle_r"],
    # volitelné pro symetrii
    "l_hip":  ["left_hip","LeftHip","l_hip","LHip","hip_l"],
    "l_knee": ["left_knee","LeftKnee","l_knee","LKnee","knee_l"],
    "l_ankle":["left_ankle","LeftAnkle","l_ankle","LAnkle","ankle_l"],
    # další (rameno apod.) můžeš přidat dle souboru
}

def _find_index(name_list: List[str], targets: List[str]) -> int | None:
    low = [s.lower() for s in name_list]
    for t in targets:
        t_low = t.lower()
        for i, s in enumerate(low):
            if t_low == s or t_low in s:
                return i
    return None

def load_factorial_json(path: str | Path, use_3d: bool = False) -> Tuple[np.ndarray, List[str], float]:
    """
    Vrací:
      data: [frames, joints, dims]  (dims=2 pro x,y; dims=3 pro x,y,z)
      joint_names: seznam jmen kloubů v pořadí os
      fps: snímková frekvence (pokud v souboru není, dá fallback 60.0)

    Podporuje dva layouty:
      1) top-level dict s klíči 'keypoints2D' / 'keypoints3D'
      2) top-level list snímků: [{ "0": { keypoints2D:[{x,y,(z),name,...}, ...], ... } }, ...]
    """
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))

    # fps – pokus o detekci, jinak fallback
    fps = 60.0
    if isinstance(obj, dict):
        for k in ("fps", "frame_rate", "framerate"):
            if k in obj and isinstance(obj[k], (int, float)):
                fps = float(obj[k]); break

    # --- varianta A: top-level dict s keypoints ---
    if isinstance(obj, dict) and ("keypoints3D" in obj or "keypoints2D" in obj):
        kpts_key = "keypoints3D" if (use_3d and "keypoints3D" in obj) else "keypoints2D"
        kpts = obj[kpts_key]

        if isinstance(kpts, list) and kpts and isinstance(kpts[0], dict) and "name" in kpts[0]:
            # struktura: list kloubů -> každý má frames
            joint_names = [j["name"] for j in kpts]
            frames = len(kpts[0]["frames"])
            dims = 3 if (use_3d and kpts_key == "keypoints3D") else 2
            data = np.zeros((frames, len(joint_names), dims), dtype=float)
            for j_idx, j in enumerate(kpts):
                for t, fr in enumerate(j["frames"]):
                    if dims == 3:
                        data[t, j_idx, :] = [fr["x"], fr["y"], fr.get("z", 0.0)]
                    else:
                        data[t, j_idx, :] = [fr["x"], fr["y"]]
            return data, joint_names, fps

        elif isinstance(kpts, list) and kpts and isinstance(kpts[0], dict) and "points" in kpts[0]:
            # struktura: list snímků -> každý má "points"
            frames = len(kpts)
            joint_names = [pt["name"] for pt in kpts[0]["points"]]
            dims = 3 if (use_3d and kpts_key == "keypoints3D") else 2
            data = np.zeros((frames, len(joint_names), dims), dtype=float)
            name_to_idx = {n: i for i, n in enumerate(joint_names)}
            for t, rec in enumerate(kpts):
                for pt in rec["points"]:
                    j_idx = name_to_idx[pt["name"]]
                    if dims == 3:
                        data[t, j_idx, :] = [pt["x"], pt["y"], pt.get("z", 0.0)]
                    else:
                        data[t, j_idx, :] = [pt["x"], pt["y"]]
            return data, joint_names, fps

    # --- varianta B: top-level list snímků jako u tebe ---
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # každý prvek je např. {"0": { "keypoints2D": [...], ...}}
        # vezmeme první dict a vybalíme jeho jediný vnitřní záznam
        first_outer = obj[0]
        if len(first_outer) != 1:
            raise ValueError(f"{p.name}: nečekaný tvar snímku (více klíčů na 1. úrovni).")
        first_key = next(iter(first_outer.keys()))
        first_frame = first_outer[first_key]

        # zkusíme 3D jen pokud je explicitně k dispozici a use_3d=True
        kpts_key = "keypoints3D" if (use_3d and "keypoints3D" in first_frame) else "keypoints2D"
        if kpts_key not in first_frame:
            raise ValueError(f"{p.name}: v rámcích chybí {kpts_key}.")

        # jména kloubů z prvního snímku
        joint_names = [pt["name"] for pt in first_frame[kpts_key]]
        dims = 3 if (use_3d and kpts_key == "keypoints3D") else 2
        frames = len(obj)
        data = np.zeros((frames, len(joint_names), dims), dtype=float)

        for t, outer in enumerate(obj):
            if len(outer) != 1:
                continue
            inner = next(iter(outer.values()))
            pts = inner.get(kpts_key, [])
            # map jméno -> index
            name_to_idx = {n: i for i, n in enumerate(joint_names)}
            for pt in pts:
                j_idx = name_to_idx.get(pt["name"])
                if j_idx is None:
                    continue
                if dims == 3:
                    data[t, j_idx, :] = [pt["x"], pt["y"], pt.get("z", 0.0)]
                else:
                    data[t, j_idx, :] = [pt["x"], pt["y"]]
        # fps fallback – občas je v každém framu, ale často není; necháme 60.0
        return data, joint_names, fps

    # --- pokud jsme se sem dostali, formát je jiný ---
    raise ValueError(f"{p.name}: nepodporovaný formát JSON (čekal jsem frames-list nebo keypoints2D/3D).")

def select_joints(data: np.ndarray, joint_names: List[str], wanted: List[str]) -> Tuple[np.ndarray, List[str]]:
    idxs = []
    used = []
    for w in wanted:
        cand = _find_index(joint_names, FACTORIAL_JOINT_ALIASES.get(w, [w]))
        if cand is not None:
            idxs.append(cand); used.append(joint_names[cand])
        else:
            # allow missing (we'll drop); warn via print (or logging)
            print(f"[pose_loader] Warning: joint '{w}' not found; skipping.")
    if not idxs:
        raise ValueError("No requested joints found in JSON.")
    return data[:, idxs, :], used
