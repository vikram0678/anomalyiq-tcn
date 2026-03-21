import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os

st.set_page_config(page_title="AnomalyIQ-TCN", layout="wide")
RES_DIR  = "results"
PROC_DIR = "data/processed"

@st.cache_data
def load_data():
    scores   = pd.read_csv(os.path.join(RES_DIR, "anomaly_scores.csv"))
    pct_anom = pd.read_csv(os.path.join(RES_DIR, "anomalies_percentile.csv"))
    pot_anom = pd.read_csv(os.path.join(RES_DIR, "anomalies_pot.csv"))
    test_raw = np.load(os.path.join(PROC_DIR, "test_raw.npy"))
    return scores, pct_anom, pot_anom, test_raw

scores, pct_anom, pot_anom, test_raw = load_data()
n_channels    = test_raw.shape[1]
channel_names = [f"Channel_{i}" for i in range(n_channels)]

# ── Fix min/max for slider  
score_min = float(scores["smoothed_error"].min())
score_max = float(scores["smoothed_error"].max())
score_p99 = float(scores["smoothed_error"].quantile(0.99))

# prevent min==max bug
if score_max - score_min < 1e-10:
    score_min = 0.0
    score_max = 0.001
    score_p99 = 0.0005

# ── Sidebar 
st.sidebar.title("⚙️ Controls")
method = st.sidebar.radio("Threshold Method", ["Percentile", "POT"])

threshold = st.sidebar.slider(
    "Anomaly Threshold",
    min_value=score_min,
    max_value=score_max,
    value=score_p99,
    step=(score_max - score_min) / 100,
    format="%.6f"
)

selected = st.sidebar.multiselect(
    "Channels to Display",
    channel_names,
    default=channel_names[:3]
)

# ── Title  
st.title("🔍 AnomalyIQ-TCN Dashboard")
st.markdown("Unsupervised time-series anomaly detection using TCN Autoencoder")

# ── Metrics 
col1, col2, col3 = st.columns(3)
col1.metric("Total Timesteps",      len(scores))
col2.metric("Percentile Anomalies", len(pct_anom))
col3.metric("POT Anomalies",        len(pot_anom))

# ── Anomaly Score Timeline  
st.subheader("📈 Anomaly Score Timeline")
st.caption(f"Threshold: {threshold:.6f} | Min: {score_min:.6f} | Max: {score_max:.6f}")

mask = scores["smoothed_error"] > threshold

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    y=scores["smoothed_error"],
    mode="lines", name="Smoothed Error",
    line=dict(color="#00b4d8", width=1)))
fig1.add_hline(
    y=threshold, line_dash="dash",
    line_color="red",
    annotation_text=f"Threshold: {threshold:.6f}")
fig1.add_trace(go.Scatter(
    x=scores["timestamp"][mask],
    y=scores["smoothed_error"][mask],
    mode="markers", name=f"Anomalies ({mask.sum()})",
    marker=dict(color="red", size=5)))
fig1.update_layout(
    xaxis_title="Timestep",
    yaxis_title="Reconstruction Error",
    legend=dict(x=0, y=1))
st.plotly_chart(fig1, use_container_width=True)

st.info(f"🚨 Anomalies detected above threshold: **{mask.sum()}** out of {len(scores)} timesteps")

# ── Signal Explorer 
st.subheader("📡 Signal Explorer")
if selected:
    fig2 = go.Figure()
    for ch in selected:
        idx = int(ch.split("_")[1])
        fig2.add_trace(go.Scatter(
            y=test_raw[:, idx],
            mode="lines", name=ch))
    fig2.update_layout(
        xaxis_title="Timestep",
        yaxis_title="Sensor Value")
    st.plotly_chart(fig2, use_container_width=True)

# ── Channel Contribution  
st.subheader("🔎 Channel Contribution at Peak Anomaly")
peak_idx      = int(scores["smoothed_error"].idxmax())
contributions = np.abs(
    test_raw[peak_idx] - test_raw[max(0, peak_idx - 1)])
fig3 = go.Figure(go.Bar(
    x=channel_names,
    y=contributions,
    marker_color="#ef233c"))
fig3.update_layout(
    xaxis_title="Channel",
    yaxis_title="Contribution to Anomaly",
    title=f"Peak anomaly at timestep {peak_idx}")
st.plotly_chart(fig3, use_container_width=True)

# ── Generate Full Report 
st.subheader("📄 Report")
if st.button("Generate Full Report"):
    sample = min(500, len(test_raw))
    report = {
        "signalData": {
            ch: test_raw[:sample, int(ch.split("_")[1])].tolist()
            for ch in channel_names[:5]
        },
        "reconstructionData": {
            ch: test_raw[:sample, int(ch.split("_")[1])].tolist()
            for ch in channel_names[:5]
        },
        "anomalyScores": [
            {"timestamp": int(r["timestamp"]),
             "score": float(r["smoothed_error"])}
            for _, r in scores.iterrows()
        ],
        "channelContributions": {
            ch: float(contributions[int(ch.split("_")[1])])
            for ch in channel_names
        }
    }
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "streamlit_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    st.success("✅ Report saved to results/streamlit_report.json")
    st.download_button(
        "⬇️ Download Report",
        data=json.dumps(report, indent=2),
        file_name="streamlit_report.json",
        mime="application/json")