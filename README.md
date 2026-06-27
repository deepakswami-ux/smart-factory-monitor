# 🏭 Smart Factory Monitor

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.20-purple?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> **Real-time ML anomaly detection dashboard for industrial machines — built with Isolation Forest and Streamlit.**

🔗 **[Live Demo → smart-factory-monitor.streamlit.app](https://smart-factory-monitor.streamlit.app)**

---

## What This Does

This project simulates a smart factory monitoring system. It ingests sensor data from 4 industrial machines in real time, runs an unsupervised ML model to detect anomalies, and displays a live dashboard with health scores, alerts, and maintenance recommendations.

Think of it as a lightweight version of what real predictive maintenance systems do in manufacturing plants — but built from scratch in Python.

---

## Features

- **Live sensor simulation** — temperature, vibration, pressure, and motor speed for 4 machines (MCH-001 to MCH-004)
- **Isolation Forest ML model** — unsupervised anomaly detection with 5% contamination rate
- **Health score (0–100)** — per machine, updated on every refresh
- **Risk classification** — Healthy / Warning / Critical with recommended actions
- **Remaining Useful Life (RUL) estimate** — based on anomaly score trends
- **Interactive Plotly charts** — sensor trend lines + anomaly score timeline
- **Health gauge** — visual indicator per machine
- **Live refresh** — one-click data update

---

## How the ML Model Works

The model uses **Isolation Forest** from scikit-learn — an unsupervised algorithm that detects anomalies without needing labelled data.

**Key idea:** Normal data points are hard to isolate (they blend in). Anomalies are easy to isolate (they stand out). The algorithm builds random decision trees and measures how many splits it takes to isolate each point. Fewer splits = more anomalous.

**Pipeline:**
1. Generate 500 historical readings per machine (train set)
2. Standardize features with `StandardScaler` so scale doesn't bias the model
3. Train `IsolationForest` with `contamination=0.05` (expects 5% anomalies)
4. On new readings: score → anomaly score (0–100) → health score → risk level

---

## Project Structure

```
smart-factory-monitor/
├── app.py                ← Streamlit dashboard (UI, charts, metrics)
├── model.py              ← AnomalyDetector class (Isolation Forest)
├── data_generator.py     ← Simulates sensor data for 4 machines
└── requirements.txt      ← Python dependencies
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/deepakswami-ux/smart-factory-monitor.git
cd smart-factory-monitor

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Streamlit | Web dashboard |
| Scikit-learn | Isolation Forest ML model |
| Plotly | Interactive charts |
| Pandas | Data manipulation |
| NumPy | Numerical operations |

---

## Machines Monitored

| Machine ID | Name | Key Sensors |
|------------|------|-------------|
| MCH-001 | Hydraulic Press | Temp, Pressure, Vibration, RPM |
| MCH-002 | CNC Milling Machine | Temp, Pressure, Vibration, RPM |
| MCH-003 | Industrial Compressor | Temp, Pressure, Vibration, RPM |
| MCH-004 | Conveyor Belt Drive | Temp, Pressure, Vibration, RPM |

---

## About

Built by **Deepak Swami** — a final-year Mechanical Engineering student learning Python and ML from scratch, with the goal of breaking into Automation and AI Engineering.

This project is part of a self-directed learning journey to bridge mechanical engineering fundamentals with modern AI/ML tools.
