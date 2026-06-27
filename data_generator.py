"""
data_generator.py
Simulates realistic industrial sensor data for 4 machines.
Injects anomalies at a 5% rate to give the ML model something to detect.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

MACHINES = {
    "MCH-001": {"name": "Hydraulic Press", "temp_mean": 75, "temp_std": 5, "vib_mean": 2.5, "vib_std": 0.3, "pres_mean": 150, "pres_std": 8, "speed_mean": 1450, "speed_std": 30},
    "MCH-002": {"name": "CNC Milling Machine", "temp_mean": 65, "temp_std": 4, "vib_mean": 3.0, "vib_std": 0.4, "pres_mean": 120, "pres_std": 6, "speed_mean": 2800, "speed_std": 50},
    "MCH-003": {"name": "Industrial Compressor", "temp_mean": 90, "temp_std": 6, "vib_mean": 4.0, "vib_std": 0.5, "pres_mean": 200, "pres_std": 10, "speed_mean": 960, "speed_std": 20},
    "MCH-004": {"name": "Conveyor Belt Drive", "temp_mean": 55, "temp_std": 3, "vib_mean": 1.5, "vib_std": 0.2, "pres_mean": 80, "pres_std": 5, "speed_mean": 500, "speed_std": 15},
}

ANOMALY_RATE = 0.05
HISTORY_POINTS = 100


def generate_single_reading(machine_id, inject_anomaly=False):
    cfg = MACHINES[machine_id]
    rng = np.random
    if inject_anomaly:
        multiplier = rng.uniform(2.5, 4.0)
        temp  = cfg["temp_mean"]  + multiplier * cfg["temp_std"]
        vib   = cfg["vib_mean"]   + multiplier * cfg["vib_std"]
        pres  = cfg["pres_mean"]  + multiplier * cfg["pres_std"]
        speed = cfg["speed_mean"] + rng.choice([-1, 1]) * multiplier * cfg["speed_std"]
    else:
        temp  = rng.normal(cfg["temp_mean"],  cfg["temp_std"])
        vib   = rng.normal(cfg["vib_mean"],   cfg["vib_std"])
        pres  = rng.normal(cfg["pres_mean"],  cfg["pres_std"])
        speed = rng.normal(cfg["speed_mean"], cfg["speed_std"])
    return {
        "machine_id": machine_id, "machine_name": cfg["name"], "timestamp": datetime.now(),
        "temperature": round(max(0, temp), 2), "vibration": round(max(0, vib), 3),
        "pressure": round(max(0, pres), 1), "motor_speed": round(max(0, speed), 0),
        "is_anomaly_injected": inject_anomaly,
    }


def generate_live_reading(machine_id):
    inject = np.random.random() < ANOMALY_RATE
    return generate_single_reading(machine_id, inject_anomaly=inject)


def generate_history(machine_id, n_points=HISTORY_POINTS):
    records = []
    now = datetime.now()
    for i in range(n_points):
        inject = np.random.random() < ANOMALY_RATE
        reading = generate_single_reading(machine_id, inject_anomaly=inject)
        reading["timestamp"] = now - timedelta(seconds=30 * (n_points - i))
        records.append(reading)
    return pd.DataFrame(records)


def get_all_machine_ids():
    return list(MACHINES.keys())


def get_machine_name(machine_id):
    return MACHINES[machine_id]["name"]
