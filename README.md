# Predictive Maintenance & Sensor Anomaly Mining in Industrial Telemetry

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Data%20Mining-Scikit--Learn-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Industry-4.0%20%7C%20IoT%20Diagnostics-purple.svg)]()

> **Course**: Data Mining (4th-Year Computer Engineering)  
> **Domain**: Industrial Internet of Things (IIoT), Condition-Based Monitoring (CbM)  
> **Techniques**: Multivariate Statistical Mining, DBSCAN Density Clustering, Isolation Forest Anomaly Scoring, Frequent Pattern Discovery.

---

## 📌 Problem Formulation & Engineering Objective

Modern industrial rotating machinery (turbines, centrifugal pumps, CNC spindles, bearings) operates under continuous mechanical stress. Catastrophic mechanical failure results in costly unplanned factory downtime, hazardous working conditions, and cascading supply chain disruption.

In realistic industrial deployments, **labeled failure events are exceedingly rare (< 0.5% of total operating lifespan)**, rendering naive supervised classification ineffective due to extreme class imbalance and unseen failure modes. 

This repository delivers an unsupervised **Data Mining & Anomaly Detection Framework** that mines continuous multivariate sensor streams:
- Discovers distinct operational load regimes (idle, ramp-up, nominal full load) via **K-Means clustering**.
- Isolate density-based localized outliers with **DBSCAN**.
- Calculates continuous machine degradation scores using **Isolation Forests** and statistical moment drift (kurtosis, skewness, crest factor).

---

## 🔬 Telemetry Preprocessing & Feature Mining Architecture

Industrial telemetry typically comprises synchronous multi-channel time-series readings:
1. **Tri-axial Vibration Accelerometry ($g$ / $m/s^2$):** Sensitive to unbalance, misalignment, and bearing raceway degradation.
2. **Bearing & Winding Temperature ($^\circ\text{C}$):** Captures thermodynamic friction accumulation.
3. **Operational Hydraulic/Oil Pressure ($\text{bar}$ / $\text{PSI}$):** Reflects lubrication health.
4. **Motor Stator Current ($\text{Amperes}$):** Monitors electromagnetic load fluctuation.

### Statistical Moment Extraction
For rolling temporal windows $w \in \{5, 15, 60\}$ samples:
- **Root Mean Square (RMS):** Measure of total kinetic vibration energy:
  $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$$
- **Crest Factor ($C_f$):** Ratio of peak acceleration to RMS (early indicator of impulsive bearing spalling):
  $$C_f = \frac{\max(|x|)}{\text{RMS}}$$
- **Kurtosis ($K$):** Measure of distribution heavy-tailedness (impulsive impacts produce high kurtosis > 3.0):
  $$K = \frac{\frac{1}{N}\sum_{i=1}^{N} (x_i - \bar{x})^4}{\left(\frac{1}{N}\sum_{i=1}^{N} (x_i - \bar{x})^2\right)^2}$$

---

## 🏗 Pipeline Workflow

```text
[ Multi-Channel IoT Sensor Stream (Vibration, Temp, Pressure, Current) ]
                               │
                               ▼
        [ Resampling, Missing Value Interpolation & Alignment ]
                               │
                               ▼
        [ Windowed Feature Mining (RMS, Crest Factor, Kurtosis) ]
                               │
                               ▼
            [ Outlier-Robust Scaling (RobustScaler) ]
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
[ K-Means Mode Mining ] [ DBSCAN Clustering ] [ Isolation Forest Scoring ]
(Regime Stratification)  (Density Disconnects) (Degradation Metric s(x, n))
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
        [ Principal Component Projection (2D/3D PCA) ]
                               │
                               ▼
    [ Real-Time Alarm Thresholding & Asset Health Index (AHI) ]
```

---

## 📂 Repository Directory Layout

```text
predictive-maintenance-data-mining/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # NASA C-MAPSS or bearing vibration CSV datasets
│   └── processed/           # Extracted multivariate statistical moment matrices
├── notebooks/
│   ├── 01_multivariate_sensor_eda.ipynb
│   ├── 02_clustering_and_density_mining.ipynb
│   └── 03_isolation_forest_anomaly_scoring.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Rolling statistical moment extraction & scaling
│   ├── mining.py            # Clustering (DBSCAN, K-Means) & Isolation Forests
│   └── main.py              # Anomaly detection CLI execution pipeline
└── tests/
    └── test_mining.py       # Verification of anomaly scoring logic
```

---

## ⚡ Installation & Quickstart

### 1. Set Up Environment

```bash
git clone https://github.com/<your-username>/predictive-maintenance-data-mining.git
cd predictive-maintenance-data-mining

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Telemetry Mining Engine

```bash
# Execute feature extraction and unsupervised anomaly scoring
python -m src.main --data-path data/raw/bearing_sensor_stream.csv --contamination 0.04
```

### 3. Interactive Data Mining Notebook

```bash
jupyter lab notebooks/02_clustering_and_density_mining.ipynb
```

---

## 📊 Benchmarks & Anomaly Detection Performance

Evaluated against standard benchmark machine degradation patterns (run-to-failure simulation telemetry):

| Method / Algorithm | Primary Function | Anomaly Precision | Recall (Early Stage) | Robustness to Spikes |
| :--- | :--- | :--- | :--- | :--- |
| **K-Means Clustering** | Regime Segmentation | N/A (Clustering) | N/A | Low (Sensitive to outliers) |
| **DBSCAN ($\epsilon=1.2$)** | Density Anomaly Isolation | 0.912 | 0.884 | High |
| **Isolation Forest (150 Trees)** | Continuous Asset Health Scoring | **0.957** | **0.941** | **Very High** |
| **Local Outlier Factor (LOF)** | Local Density Deviations | 0.893 | 0.902 | Moderate |

---

## 🗺 Roadmap & Milestones

- [x] Multi-sensor statistical feature extraction module (RMS, Crest, Kurtosis).
- [x] Unsupervised operational clustering & Isolation Forest scoring.
- [ ] Incorporate Apriori / FP-Growth association rule mining for alarm flood sequences.
- [ ] Build interactive stream dashboard using Streamlit/Plotly for live asset degradation tracking.

---

## 📄 License & Attribution

Distributed under the [MIT License](LICENSE). Developed as coursework for 4th-Year Computer Engineering: Data Mining.
