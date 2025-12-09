# main.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from io_pkg.pose_loader import load_factorial_json, select_joints
from pre.preprocessor import center_on_pelvis, scale_by_leg_length, deriv_savgol
from gait.step_detector import detect_steps_from_ankle_y, resample_step
from features.feature_maker import flatten_xyz, make_step_features_xy
from features.spd_geom import SPDGeom, spd_from_features, spd_sequence, smooth_length, avg_step_velocity, \
    frechet_variance, pairwise_dist
from features.embedder import umap_from_distance

### NOVÉ: Import funkce pro Euklidovské metriky
from features.metrics import calculate_euclidean_metrics

# --- konfigurace ---
JOINTS_RIGHT = ["r_hip", "r_knee", "r_ankle"]  # můžeš přidat "r_toe"
AXES_2D = (0, 1)  # x,y
USE_SEQ = True  # True: SPD sekvence → Smooth; False: 1 SPD/krok


def run_spd_pipeline(path_json: str):
    # 1) load
    data, names, fps = load_factorial_json(path_json, use_3d=False)
    print(f"Frames: {data.shape[0]}  | fps: {fps:.2f}  | duration: {data.shape[0] / fps:.2f}s")

    # 2) center and scale
    data = center_on_pelvis(data, names)
    data, scale = scale_by_leg_length(data, names)
    # 3) select joints (right leg)
    data_sel, used_names = select_joints(data, names, JOINTS_RIGHT)
    # 4) flatten to [T, J*2]
    XY = flatten_xyz(data_sel, axes=AXES_2D)
    # 5) detect steps on ankle y  (najdeme skutečný index kotníku v used_names)
    print("Detected joints:", used_names)  # jednorázově pro kontrolu

    names_lower = [n.lower() for n in used_names]
    ankle_idx = None
    for i, n in enumerate(names_lower):
        if "ankle" in n or "kotnik" in n:
            ankle_idx = i
            break

    # fallback: když v souboru není „ankle“, zkus „heel“ nebo „foot“
    if ankle_idx is None:
        for i, n in enumerate(names_lower):
            if "heel" in n or "foot" in n or "toe" in n:
                ankle_idx = i
                print(f"[step] ankle fallback -> using joint '{used_names[i]}'")
                break

    if ankle_idx is None:
        raise RuntimeError(
            f"Nenašel jsem kotník/heel/foot v {used_names}. "
            f"Uprav prosím aliasy nebo vyber jiný kloub pro detekci kroků."
        )

    axis_y = 1 if len(AXES_2D) >= 2 else 0
    ankle_y_col = ankle_idx * len(AXES_2D) + axis_y

    steps = detect_steps_from_ankle_y(XY[:, ankle_y_col], fps=fps)
    if not steps:
        print("No steps detected.")
        return

    # 6) per-step features & SPD
    spd_mats = []  # 1 SPD per step (for clustering/UMAP or Var_R)
    smooth_vals = []  # per-step Smooth (if USE_SEQ)
    vbar_vals = []
    geom = None

    ### NOVÉ: Seznamy pro ukládání Euklidovských výsledků
    euclid_smooth_list = []
    euclid_v_list = []
    euclid_var_list = []

    for (a, b) in steps:
        step_xy = resample_step(XY, a, b, num=101)  # [101, J*2]

        ### NOVÉ: Výpočet Euklidovských metrik pro tento konkrétní krok
        # Používáme stejná data (step_xy) jako pro Riemanna, takže srovnání je férové
        e_s, e_v, e_var = calculate_euclidean_metrics(step_xy)
        euclid_smooth_list.append(e_s)
        euclid_v_list.append(e_v)
        euclid_var_list.append(e_var)

        # --- Původní Riemannovská část ---
        feat = make_step_features_xy(step_xy, fps=fps, use_z=False)  # [101, D]
        if geom is None:
            geom = SPDGeom(dim=feat.shape[1])  # d = J*2*(pos+vel)

        if USE_SEQ:
            seq = spd_sequence(feat, win=11)  # list of [d,d]
            smooth_vals.append(smooth_length(seq, geom))
            vbar_vals.append(avg_step_velocity(seq, geom))
            # také si ulož 1 SPD na krok (kovariance přes celou fázi)
            spd_mats.append(spd_from_features(feat))
        else:
            C = spd_from_features(feat)
            spd_mats.append(C)

    # 7) variability across steps (Fréchet variance)
    var_r = frechet_variance(spd_mats, geom) if len(spd_mats) >= 2 else 0.0

    # 8) optional: UMAP map of steps
    if len(spd_mats) >= 3:
        D = pairwise_dist(spd_mats, geom)
        emb = umap_from_distance(D, n_components=2)
        print(f"UMAP embedding shape: {emb.shape}")
    else:
        emb = None
        print("UMAP přeskočen (málo kroků).")

    # 9) print/report minimal
    print(f"Steps: {len(steps)}")

    # Výpis Riemannovských metrik
    if smooth_vals:
        print(f"Riemann Smooth (median): {np.median(smooth_vals):.3f}  |  v_bar (median): {np.median(vbar_vals):.3f}")
    print(f"Riemann Var_R (across steps): {var_r:.3f}")

    # Výpis Euklidovských metrik (Medián)
    print("-" * 30)
    print("--- EUCLIDEAN BASELINE (Median) ---")
    if euclid_smooth_list:
        print(f"Euclid Smooth (Path Len): {np.median(euclid_smooth_list):.3f}")
        print(f"Euclid v_bar (Mean Vel):  {np.median(euclid_v_list):.3f}")
        print(f"Euclid Var (Total Var):   {np.median(euclid_var_list):.3f}")
    print("-" * 30)


if __name__ == "__main__":
    # přizpůsob si cestu na svoje soubory
    run_spd_pipeline("data/run_fast_100%.json")
