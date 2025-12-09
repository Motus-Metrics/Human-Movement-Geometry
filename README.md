# Riemannian Gait Analysis Framework

A Python computational tool for analyzing human gait dynamics using Riemannian geometry and manifold learning on Symmetric Positive Definite (SPD) matrices.

This repository contains the source code and proof-of-concept data for the research paper comparing linear (Euclidean) and non-linear (Riemannian) metric sensitivity in biomechanical analysis.

## Associated Publication
* **Title:** Riemannian Manifold Representation of Human Motion: A Framework for Factorial Biomechanics Analysis
* **Status:** Manuscript in preparation
* **Link:** [Link to arXiv or PDF will be added here]

## Overview
Current Euclidean approaches often fail to capture the intrinsic non-linear geometric structure of human movement. This framework transforms raw kinematic data (pose estimation JSONs) into a Riemannian manifold to provide physically meaningful metrics for diagnostics and machine learning.

Key capabilities:
* Transformation of joint coordinates into covariance matrices (SPD manifold).
* Calculation of Geodesic Distance, Riemann Variance, and Smoothness.
* Direct comparison with baseline Euclidean metrics (velocity, path length).
* Visualization of metric divergence across different gait speeds.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/TommiBu/Riemannian-gait-analysis](https://github.com/TommiBu/Riemannian-gait-analysis)
   cd riemannian-gait-analysis
2. Install dependencies:
   pip install -r requirements.txt

## License

* **Source Code:** Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
You are free to use the code for your own research (including commercial), but the specific dataset and the content of the scientific paper cannot be used for commercial purposes without permission.
