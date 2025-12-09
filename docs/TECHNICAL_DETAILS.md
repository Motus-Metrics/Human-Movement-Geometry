## Geometric Gait Analysis Project
This project analyzes gait data (human motion sequences) using Riemannian geometry on Symmetric Positive Definite (SPD) matrices. It processes skeletal pose data to compute stability and variability metrics, comparing a geometric approach against a standard Euclidean baseline.

## Directory Structure
data/: Input data files in JSON format (e.g., run_fast_100%.json, run_slow_100%.json).
features/: Core mathematical logic.
spd_geom.py: Functions for Riemannian geometry (SPD matrices, geodesics, Fréchet variance).
embedder.py: Dimensionality reduction (UMAP) for visualization.
metrics.py: Ca# Technical Documentation: Riemannian Gait Analysis

This document outlines the computational framework, mathematical principles, and software architecture used in the **Riemannian Gait Analysis** project. The goal of this framework is to quantify human motion stability using non-Euclidean geometry, specifically by mapping gait cycles onto the manifold of Symmetric Positive Definite (SPD) matrices.

## 1. Mathematical Framework

Traditional biomechanics relies on Euclidean metrics (e.g., position variance) which often treat joint correlations linearly. This project employs **Riemannian Geometry** to capture the intrinsic non-linear structure of human movement.

### 1.1. Data Representation (SPD Manifold)
For a given gait cycle (step), we represent the movement not as a sequence of vectors, but as a **Covariance Matrix** ($C$).
Given a data matrix $X \in \mathbb{R}^{T \times N}$ (where $T$ is time frames and $N$ is degrees of freedom/joints):

$$C = \frac{1}{T-1} (X - \bar{X})^T (X - \bar{X})$$

Since $C$ is a Symmetric Positive Definite (SPD) matrix, it does not lie in a flat vector space but on a curved Riemannian manifold $\mathcal{M} = Sym_d^+$.

### 1.2. The Log-Euclidean Metric
To perform statistical operations (like computing mean or variance) on this curved space, we utilize the **Log-Euclidean framework**. This maps points from the manifold to the tangent space at the identity matrix via the matrix logarithm:

$$L = \log_m(C) = U \log(\Sigma) U^T$$

Where $C = U \Sigma U^T$ is the eigendecomposition. In this tangent space, standard Euclidean statistics can be validly applied.

---

## 2. Analysis Pipeline

The analysis logic is centrally controlled by `main.py` and proceeds in five sequential stages:

### Step 1: Data Loading & Preprocessing
* **Input:** Factorial JSON files containing skeletal pose data.
* **Normalization:**
    1.  **Centering:** The skeleton is translated so the pelvis (root) is at $(0,0,0)$.
    2.  **Scaling:** All coordinates are normalized by the subject's leg length to ensure metric invariance across different body heights.

### Step 2: Gait Segmentation
* **Method:** Zero-crossing and peak detection on the vertical velocity ($v_y$) of the ankle joint.
* **Output:** A list of `(start_frame, end_frame)` tuples representing individual steps.

### Step 3: Feature Extraction
For each detected step, raw coordinates of selected joints (e.g., Right Hip, Knee, Ankle) are extracted.
* **Euclidean Features:** Raw Cartesian coordinates $(x, y)$.
* **Riemannian Features:** The raw window is converted into a Covariance Matrix ($C$) and subsequently projected to the tangent space ($L$).

### Step 4: Metric Calculation
The core comparison is performed by computing parallel metrics in both spaces:

| Metric Category | Euclidean Baseline (Linear) | Riemannian Approach (Non-Linear) |
| :--- | :--- | :--- |
| **Input Data** | Raw Trajectory Vectors ($X$) | SPD Matrices ($C$) / Tangent Vectors ($L$) |
| **Smoothness** | Cumulative path length: $\sum \|v_t\|$ | Geodesic length of sequence on Manifold |
| **Velocity** | Mean scalar velocity: $\frac{1}{T}\sum v_t$ | Average velocity on Manifold |
| **Variability** | Total Variation: $Trace(Cov(X))$ | **Fréchet Variance:** Dispersion around the geometric mean |

### Step 5: Visualization
* **UMAP:** The computed SPD matrices are flattened and passed to UMAP (Uniform Manifold Approximation and Projection) to visualize the topological structure of steps in 2D space.

---

## 3. Directory Structure & Modules

The codebase is organized to separate data handling, geometric logic, and execution.

```text
├── main.py                 # Entry point: Runs the full pipeline
├── data/                   # Input JSON datasets
│   ├── run_slow_100%.json
│   ├── run_mid_100%.json
│   └── run_fast_100%.json
├── features/               # Core mathematical logic
│   ├── spd_geom.py         # Riemannian geometry (Log-map, Fréchet mean)
│   ├── metrics.py          # Euclidean baseline calculations
│   ├── embedder.py         # UMAP dimensionality reduction
│   └── feature_maker.py    # Formatting raw data into matrices
├── gait/
│   └── step_detector.py    # Logic for segmenting steps from video
├── pre/
│   └── preprocessor.py     # Normalization (Pelvis centering, Scaling)
└── io_pkg/
    └── pose_loader.py      # JSON parsing utilities
```

## 4. Configuration

  Key parameters can be adjusted directly in `main.py`:

  * `JOINTS_RIGHT`: List of joints to include in the analysis.
    * **Default:** `['r_hip', 'r_knee', 'r_ankle']`

  * `USE_SEQ`: Controls the temporal resolution of the manifold mapping.
    * `True`: Treats a step as a **sequence** of smaller SPD matrices. This mode is required for calculating dynamic metrics like **Smoothness** (geodesic length) and **Velocity** on the manifold.
    * `False`: Collapses the entire step into **one** global SPD matrix. This approach is optimized for tasks like **Clustering** (UMAP) and calculating simple **Fréchet Variance**.
