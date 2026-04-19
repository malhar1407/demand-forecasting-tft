import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.models.loader import load_model_and_data, get_test_predictions

st.set_page_config(page_title="Forecasts", page_icon="📈", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 17px; }
    .metric-label { font-size: 16px !important; }
    .metric-value { font-size: 28px !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Demand Forecasts")
st.markdown("##### 6-month ahead demand predictions for each Agency–SKU combination")

_, _, train, val, test = load_model_and_data()
actual, quantile_preds, test_dataset = get_test_predictions()

agencies = sorted(test["agency"].unique())
skus     = sorted(test["sku"].unique())

col1, col2 = st.columns(2)
agency = col1.selectbox("Agency", agencies)
sku    = col2.selectbox("SKU", skus)

# Map to sample index
decoded     = test_dataset.decoded_index
sample_mask = (decoded["agency"] == agency) & (decoded["sku"] == sku)
sample_indices = np.where(sample_mask)[0]

full = pd.concat([train, val, test]).reset_index(drop=True)

if len(sample_indices) == 0:
    st.warning("No predictions found for this combination.")
    st.stop()

si   = sample_indices[0]
act  = actual[si]
q02  = quantile_preds[si, :, 0]
q10  = quantile_preds[si, :, 1]
q50  = quantile_preds[si, :, 3]
q90  = quantile_preds[si, :, 5]
q98  = quantile_preds[si, :, 6]

horizon = len(act)
first_pred_time_idx = decoded.loc[si, "time_idx_first_prediction"]
group_df   = full[(full["agency"] == agency) & (full["sku"] == sku)]
start_date = group_df.loc[group_df["time_idx"] == first_pred_time_idx, "date"].values[0]
dates      = pd.date_range(start_date, periods=horizon, freq="MS")

hist        = group_df.sort_values("time_idx").tail(24 + horizon)
hist_dates  = pd.to_datetime(hist["date"])
hist_volume = hist["volume"].values

# --- Chart ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hist_dates, y=hist_volume, name="Historical Demand",
    line=dict(color="#888", width=1.5),
    hovertemplate="<b>Historical</b><br>%{x|%b %Y}: %{y:,.0f} units<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=np.concatenate([dates, dates[::-1]]),
    y=np.concatenate([q02, q98[::-1]]),
    fill="toself", fillcolor="rgba(99,110,250,0.1)",
    line=dict(width=0), name="95% Interval",
    hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=np.concatenate([dates, dates[::-1]]),
    y=np.concatenate([q10, q90[::-1]]),
    fill="toself", fillcolor="rgba(99,110,250,0.25)",
    line=dict(width=0), name="80% Interval",
    hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=dates, y=q50, name="Forecast (q0.5)",
    line=dict(color="#636EFA", width=2.5),
    hovertemplate="<b>Forecast (q0.5)</b><br>%{x|%b %Y}: %{y:,.0f} units<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=dates, y=act, name="Actual Demand",
    line=dict(color="#EF553B", width=2, dash="dot"),
    hovertemplate="<b>Actual</b><br>%{x|%b %Y}: %{y:,.0f} units<extra></extra>",
))

fig.update_layout(
    xaxis_title="Month", yaxis_title="Demand Volume (units)",
    legend=dict(orientation="h", y=-0.22, font=dict(size=14)),
    hovermode="x unified", height=460,
    font=dict(size=14),
)
st.plotly_chart(fig, use_container_width=True)

# --- Legend explanation ---
with st.expander("📖 How to read this chart", expanded=False):
    st.markdown("""
| Element | What it means |
|---|---|
| **Historical Demand** (grey line) | Actual past sales used to train the model |
| **Forecast (q0.5)** (blue line) | The model's best guess — demand is equally likely to be above or below this line |
| **80% Interval** (darker band) | The model is 80% confident actual demand will fall in this range |
| **95% Interval** (lighter band) | A wider safety net — 95% confident actual demand lands here. A wider band means more uncertainty |
| **Actual Demand** (red dotted) | What demand really was — only visible for the test period the model never saw during training |
""")

st.divider()

# --- Accuracy summary ---
st.markdown("#### Forecast Accuracy Summary")

mae   = np.mean(np.abs(act - q50))
rmse  = np.sqrt(np.mean((act - q50) ** 2))
mape  = np.mean(np.abs(act - q50) / (np.abs(act) + 1e-8)) * 100
within_80 = int(np.sum((act >= q10) & (act <= q90)))
within_95 = int(np.sum((act >= q02) & (act <= q98)))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("MAE", f"{mae:,.0f} units", help="Mean Absolute Error — average gap between forecast and actual demand")
c2.metric("RMSE", f"{rmse:,.0f} units", help="Root Mean Squared Error — penalises large misses more heavily")
c3.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error — average % miss relative to actual demand")
c4.metric("Within 80% Interval", f"{within_80}/{horizon} months", help="How many months actual demand fell inside the 80% prediction band")
c5.metric("Within 95% Interval", f"{within_95}/{horizon} months", help="How many months actual demand fell inside the 95% prediction band")
