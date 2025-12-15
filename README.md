# Riemannian Gait Analysis Framework

**Riemannian vs. Euclidean Frameworks in Gait Kinematics**

A Python computational tool for analyzing human gait dynamics using Riemannian geometry and manifold learning on Symmetric Positive Definite (SPD) matrices.

This repository accompanies the research paper listed below. It serves as a **Proof of Concept** demonstrating that mapping biomechanical data onto a curved manifold captures efficiency metrics (stabilization at high speeds) that standard Euclidean methods miss.

## Associated Publication
* **Title:** Riemannian vs. Euclidean Representation of Gait Kinematics: A Comparative Analysis
* **Status:** Manuscript in preparation
* **Read the Paper:** [https://doi.org/10.48550/arXiv.2512.09158]

---

## Getting Started

### 1. Installation
Clone the repository and install the required dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/MotusMetrics/hmg.git

# 2. Navigate to the directory
cd hmg

# 3. Install dependencies
pip install -r requirements.txt
```
### 2. Data Preparation

The current pipeline is optimized for JSON outputs from Factorial Biomechanics (or similar pose estimation tools).

Record/Analyze: Process your gait video using Factorial Biomechanics.

Export: Save the analysis results as a .json file.

Place: Move the JSON file into the data/ directory of this repository.

Run: Update the filename in main.py or run the script.

Note for Developers: If you use different motion capture software (e.g., OpenPose, Vicon), you can easily adapt the io_pkg/pose_loader.py module to parse your specific JSON structure.

### 3. Usage

Run the main analysis pipeline:
```bash
python main.py
```

## Modularity & Customization
This tool was built with flexibility in mind. Researchers are encouraged to modify the code to fit their specific needs:

Change Joints: Modify the JOINTS_RIGHT list in main.py to track different body parts (e.g., left leg, full body).

Adjust Metrics: The features/ directory contains modular scripts for both Riemannian (SPD) and Euclidean metric calculations.

Visualization: You can tweak UMAP parameters in features/embedder.py to explore different manifold projections.

## Community & Feedback
This is an open research project.

I created this framework for biomechanists, data scientists, and anyone interested in the intersection of geometry and human motion. Since this is a proof of concept:

Review is welcome: If you spot mathematical inconsistencies or coding improvements, please open an Issue or Pull Request.

Fork it: Feel free to fork this repository and experiment with your own datasets.

Validate: I encourage the community to test this on bilateral data or pathological gait patterns.

## Results Preview
The tool generates a comparison of variability trends. The results demonstrate that while Euclidean variance increases linearly with speed, Riemannian variance captures the non-linear biomechanical efficiency (stabilization) observed in sprinting.

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

**Open Source:** You are free to use, modify, and distribute this software, provided that any derivative work is also released as open-source under the same GPLv3 license.
**Proprietary Use:** If you wish to use this software in a **closed-source commercial product** (without disclosing your source code), you must obtain a **Commercial License**.

For commercial licensing inquiries, please contact: **Thomas Boozek** via [LinkedIn/Email].

See the [LICENSE](LICENSE) file for the full text.

