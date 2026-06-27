"""
model.py - Anomaly detection using Isolation Forest (scikit-learn).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from data_generator import generate_history, get_all_machine_ids

FEATURE_COLS = ["temperature", "vibration", "pressure", "motor_speed"]


class AnomalyDetector:
    def __init__(self, machine_id, contamination=0.05):
        self.machine_id = machine_id
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
        self.is_trained = False

    def train(self, n_history=500):
        df = generate_history(self.machine_id, n_points=n_history)
        X = df[FEATURE_COLS].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True

    def predict(self, reading):
        if not self.is_trained:
            raise RuntimeError(f"Model for {self.machine_id} has not been trained yet.")
        X = np.array([[reading["temperature"], reading["vibration"], reading["pressure"], reading["motor_speed"]]])
        X_scaled = self.scaler.transform(X)
        raw_score = self.model.score_samples(X_scaled)[0]
        anomaly_score = self._raw_to_anomaly_score(raw_score)
        health_score = round(100 - anomaly_score, 1)
        is_anomaly = self.model.predict(X_scaled)[0] == -1
        risk_level, action, rul = self._classify_risk(health_score)
        return {"anomaly_score": round(anomaly_score, 1), "health_score": health_score,
                "is_anomaly": is_anomaly, "risk_level": risk_level,
                "recommended_action": action, "rul_estimate": rul}

    def _raw_to_anomaly_score(self, raw_score):
        clipped = np.clip(raw_score, -0.6, 0.2)
        normalized = (0.2 - clipped) / (0.2 - (-0.6))
        return round(float(normalized * 100), 1)

    def _classify_risk(self, health_score):
        if health_score >= 70:
            return ("Healthy", "No action needed. Continue routine monitoring.", "> 30 days")
        elif health_score >= 40:
            return ("Warning", "Schedule preventive maintenance within 7 days.", "7-30 days")
        else:
            return ("Critical", "STOP machine immediately. Inspect for mechanical failure.", "< 7 days")


_detectors = {}

def get_detector(machine_id):
    if machine_id not in _detectors:
        detector = AnomalyDetector(machine_id)
        detector.train()
        _detectors[machine_id] = detector
    return _detectors[machine_id]


def train_all_models():
    for mid in get_all_machine_ids():
        get_detector(mid)
