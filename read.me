# Geometric Gait Manifold

Short pipeline to derive geometric, low-dimensional representations of running gait from 2D joint trajectories and compute simple coordination/smoothness metrics.

## What this does
- Loads 2D keypoint time series (per video) exported as JSON.
- Recenters trajectories on the pelvis, normalizes by leg length, smooths.
- Detects steps from ankle vertical motion; resamples each step to 0–100% phase.
- Builds a per-step “fingerprint” vector and embeds all steps via PCA (optionally UMAP).
- Reports basic metrics (currently: inter-step variability in latent space).
- Prints quick sanity stats (frames, estimated fps, number of steps, leg length, variability).

## Why this is useful
- Steps from the same runner typically lie on a smooth, low-dimensional manifold (a “style map”).
- Simple geometric metrics in that latent space can capture smoothness/consistency better than basic kinematics alone.

## Current status
- Working end-to-end PCA pipeline and one metric: variability of steps in PCA space.
- Concept/prototype quality; parameters are conservative defaults.
- FPS may be subsampled in source JSON; step events are detected from vertical ankle minima.

## What I want to do next
- Per-step latent trajectory length and curvature over 0–100% phase.
- Plots: overlay of normalized joint trajectories; 2D “style map” of steps.
- Baselines (cadence, knee angle profile) and a quick A/B separability test (logistic regression).
- Optional nonlinear embedding (UMAP) once metrics and visuals are stable.

## Data flow (how to use)
1) Record a side-view running video (phone, steady light), e.g. 60 fps.
2) Run a pose-extraction app/tool (e.g., Fractional or similar) to export joint trajectories as JSON  
   (time, 2D coordinates for at least pelvis/hip/knee/ankle/shoulder).
3) Place your JSON files in `my_gait_project/data/`.
4) Install dependencies in a Python 3.10 environment:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt
   ```
5) Run the pipeline:
   ```bash
   cd my_gait_project
   # batch over all JSONs in data/
   python main.py -i data
   # or a single file
   python main.py -i data/run_mid_data.json
   ```
6) Read the console output (for each file):
   - `frames`: number of frames parsed
   - `fs`: estimated sampling rate (Hz) from timestamps
   - `steps`: detected steps
   - `leg_len`: median hip–ankle distance before scaling
   - `varPCA_SD`: inter-step variability in PCA space (lower ≈ more consistent)

## Repository layout
```
my_gait_project/
  data_models.py
  main.py
  data/                  # JSON inputs go here
  io_pkg/pose_loader.py
  pre/preprocessor.py
  gait/step_detector.py
  features/feature_maker.py
  features/embedder.py
  features/metrics.py
  eval/evaluate.py       # not yet wired into main flow
  viz/                   # reserved for plots
```

## Transparency and attribution
- Concept, experimental design, and application to my running data are my own.
- Some scaffolding and code snippets were produced with AI assistance and then adapted/verified by me.
- This repository is a research concept; I would appreciate peer review on correctness, assumptions, and whether this geometric-manifold approach is appropriate for the intended comparisons (A vs. B).

## Notes and limitations
- JSON structure must include timestamps and named keypoints (e.g., left_hip, left_knee, left_ankle, left_shoulder). The loader can be adapted to other schemas.
- If timestamps are sparse (e.g., subsampled from 60 fps), step detection may need adjusted parameters (`--min-step-s`).
- Results are not clinical measures; they are exploratory metrics for research.

## License
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**  
This allows sharing and adaptation for non-commercial purposes, as long as credit is given to the original author.
