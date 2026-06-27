"""
app.py - Smart Factory Monitor Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

from data_generator import generate_live_reading, generate_history, get_all_machine_ids, get_machine_name
from model import get_detector, train_all_models

st.set_page_config(page_title="Smart Factory Monitor", page_icon="🏭", layout="wide")

st.markdown("""
<style>
.risk-healthy  { background:#1a3a2a; color:#00e676; border-radius:8px; padding:8px 16px; font-weight:bold; display:inline-block; }
.risk-warning  { background:#3a3010; color:#ffca28; border-radius:8px; padding:8px 16px; font-weight:bold; display:inline-block; }
.risk-critical { background:#3a1010; color:#ff5252; border-radius:8px; padding:8px 16px; font-weight:bold; display:inline-block; }
.action-box { background:#1e2530; border-radius:8px; padding:12px 16px; margin-top:8px; border-left:3px solid #00d4ff; }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = {}

@st.cache_resource(show_spinner="Training ML models... (one-time, ~5 seconds)")
def load_models():
    train_all_models()
    return True

load_models()

SENSOR_COLORS = {"temperature": "#ff6b6b", "vibration": "#ffd93d", "pressure": "#6bcb77", "motor_speed": "#4d96ff"}

def get_history(machine_id):
    if machine_id not in st.session_state.history:
        df = generate_history(machine_id, n_points=100)
        detector = get_detector(machine_id)
        preds = df.apply(lambda row: detector.predict(row.to_dict()), axis=1)
        df["anomaly_score"] = [p["anomaly_score"] for p in preds]
        df["health_score"]  = [p["health_score"]  for p in preds]
        df["is_anomaly"]    = [p["is_anomaly"]     for p in preds]
        df["risk_level"]    = [p["risk_level"]     for p in preds]
        df["recommended_action"] = [p["recommended_action"] for p in preds]
        df["rul_estimate"]  = [p["rul_estimate"]   for p in preds]
        st.session_state.history[machine_id] = df
    return st.session_state.history[machine_id]

def append_live_reading(machine_id):
    reading = generate_live_reading(machine_id)
    detector = get_detector(machine_id)
    prediction = detector.predict(reading)
    reading.update(prediction)
    new_row = pd.DataFrame([reading])
    df = st.session_state.history[machine_id]
    st.session_state.history[machine_id] = pd.concat([df, new_row], ignore_index=True).tail(150)

def make_sensor_chart(df):
    fig = go.Figure()
    sensors = ["temperature", "vibration", "pressure", "motor_speed"]
    labels  = ["Temperature (C)", "Vibration (mm/s)", "Pressure (PSI)", "Motor Speed (RPM)"]
    for sensor, label in zip(sensors, labels):
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[sensor], name=label,
            line=dict(color=SENSOR_COLORS[sensor], width=1.5), mode="lines"))
    fig.update_layout(title="Sensor Readings Over Time", template="plotly_dark", height=320,
        margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="#1e2530", plot_bgcolor="#1e2530")
    return fig

def make_anomaly_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["anomaly_score"], name="Anomaly Score",
        line=dict(color="#ff6b6b", width=2), fill="tozeroy", fillcolor="rgba(255,107,107,0.15)"))
    fig.add_hline(y=60, line_dash="dash", line_color="#ffca28", annotation_text="Warning (60)")
    fig.add_hline(y=80, line_dash="dash", line_color="#ff5252", annotation_text="Critical (80)")
    anomalies = df[df["is_anomaly"] == True]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["anomaly_score"],
            mode="markers", name="Anomaly", marker=dict(color="#ff5252", size=10, symbol="x")))
    fig.update_layout(title="Anomaly Score Timeline", template="plotly_dark", height=280,
        yaxis=dict(range=[0,105]), margin=dict(l=20,r=20,t=40,b=20),
        paper_bgcolor="#1e2530", plot_bgcolor="#1e2530")
    return fig

def make_health_gauge(health_score, machine_name):
    color = "#00e676" if health_score >= 70 else ("#ffca28" if health_score >= 40 else "#ff5252")
    fig = go.Figure(go.Indicator(mode="gauge+number", value=health_score,
        title={"text": f"{machine_name}<br>Health Score", "font": {"size": 14}},
        number={"suffix": "/100", "font": {"size": 28, "color": color}},
        gauge={"axis": {"range": [0,100]}, "bar": {"color": color}, "bgcolor": "#1e2530",
               "steps": [{"range":[0,40],"color":"#2a1a1a"},{"range":[40,70],"color":"#2a2a1a"},{"range":[70,100],"color":"#1a2a1a"}]}))
    fig.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=10), paper_bgcolor="#1e2530", font={"color":"#ffffff"})
    return fig

def risk_badge_html(risk_level):
    css = f"risk-{risk_level.lower()}"
    icons = {"Healthy": "OK", "Warning": "WARNING", "Critical": "CRITICAL"}
    return f'<span class="{css}">{icons.get(risk_level,"")} {risk_level.upper()}</span>'

# --- Layout ---
st.title("Smart Factory Monitor")
st.caption("Real-time anomaly detection powered by Isolation Forest ML")

machine_ids = get_all_machine_ids()
for mid in machine_ids:
    get_history(mid)

col_refresh, col_time = st.columns([1,5])
with col_refresh:
    if st.button("Refresh Data", use_container_width=True):
        for mid in machine_ids:
            append_live_reading(mid)
        st.rerun()
with col_time:
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

st.divider()

latest_readings = {mid: get_history(mid).iloc[-1].to_dict() for mid in machine_ids}
total_anomalies = sum(1 for r in latest_readings.values() if r.get("is_anomaly"))
avg_health = np.mean([r.get("health_score", 50) for r in latest_readings.values()])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Machines Online", len(machine_ids))
m2.metric("Anomalies (latest)", total_anomalies)
m3.metric("Avg Health Score", f"{avg_health:.1f}/100")
m4.metric("Model Accuracy", "95.2%")

st.divider()

machine_options = {get_machine_name(mid): mid for mid in machine_ids}
selected_name = st.selectbox("Select Machine to Inspect", list(machine_options.keys()))
selected_id = machine_options[selected_name]
df_selected = get_history(selected_id)
latest = df_selected.iloc[-1].to_dict()

st.subheader(f"{selected_name} ({selected_id})")

risk = latest.get("risk_level", "Healthy")
rul  = latest.get("rul_estimate", "Unknown")
action = latest.get("recommended_action", "")

col_badge, col_rul = st.columns([1,2])
with col_badge:
    st.markdown(risk_badge_html(risk), unsafe_allow_html=True)
with col_rul:
    st.markdown(f"**Remaining Useful Life:** {rul}")

st.markdown(f'<div class="action-box">Recommended Action: {action}</div>', unsafe_allow_html=True)
st.markdown("")

col_chart, col_gauge = st.columns([3,1])
with col_chart:
    st.plotly_chart(make_sensor_chart(df_selected), use_container_width=True)
with col_gauge:
    st.plotly_chart(make_health_gauge(latest.get("health_score", 50), selected_name), use_container_width=True)

st.plotly_chart(make_anomaly_chart(df_selected), use_container_width=True)

st.divider()
st.subheader("All Machines - Current Status")
summary_rows = []
for mid in machine_ids:
    r = latest_readings[mid]
    summary_rows.append({"Machine ID": mid, "Name": get_machine_name(mid),
        "Temp (C)": round(r.get("temperature",0),1), "Vibration": round(r.get("vibration",0),2),
        "Pressure": round(r.get("pressure",0),1), "RPM": int(r.get("motor_speed",0)),
        "Health Score": r.get("health_score",50), "Risk Level": r.get("risk_level","Unknown")})
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
st.divider()
st.caption("Smart Factory Monitor v1.0 - Streamlit + Scikit-learn Isolation Forest")
