import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.models.loader import load_model_and_data, get_test_predictions

st.set_page_config(page_title="Bullwhip Analysis", page_icon="🌊", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 17px; }
    p, li, .stMarkdown { font-size: 16px !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Bullwhip Effect Analysis")
st.markdown("##### How different ordering strategies reduce supply chain variance amplification")

st.markdown("""
The **bullwhip effect** is the tendency for order quantities to become increasingly volatile as you move
upstream in a supply chain — even when end-customer demand is relatively stable.
A **bullwhip ratio > 1** means orders are more volatile than actual demand (bad).
A **ratio < 1** means orders are smoother than demand (ideal).

The TFT model produces 7 quantile forecasts per period, which enables smarter ordering policies
that go beyond simply reacting to the latest demand signal.
""")

_, _, train, val, test = load_model_and_data()
actual, quantile_preds, test_dataset = get_test_predictions()

QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

full = pd.concat([train, val, test]).sort_values(["agency", "sku", "date"]).reset_index(drop=True)
full["naive_pred"] = full.groupby(["agency", "sku"])["volume"].shift(12)
test_naive = full[full["time_idx"] >= test["time_idx"].min()].dropna(subset=["naive_pred"]).reset_index(drop=True)

n            = min(len(actual.flatten()), len(test_naive))
actual_a     = actual.flatten()[:n]
naive_pred_a = test_naive["naive_pred"].values[:n]
q_flat       = quantile_preds.reshape(-1, 7)[:n]


def bullwhip(demand, orders):
    orders = np.maximum(orders, 0)
    return np.var(orders) / np.var(demand)


def smooth(arr, a):
    s = np.zeros_like(arr)
    s[0] = arr[0]
    for t in range(1, len(arr)):
        s[t] = a * arr[t] + (1 - a) * s[t - 1]
    return s


naive_ratio = bullwhip(actual_a, naive_pred_a)

# --- Sidebar ---
st.sidebar.header("Policy Settings")

st.sidebar.markdown("""
**Choose an ordering policy:**

- **Point Forecast (q0.5)** — Order exactly what the model's median forecast says. Simple and accurate, but still reactive to forecast swings.

- **Conservative Quantile** — Order at a lower quantile (e.g. q0.25), meaning you intentionally order less than the median. Reduces over-ordering and variance, but risks stockouts if demand is higher than expected.

- **Smoothed Orders** — Apply exponential smoothing to the median forecast before placing orders. The smoothing factor α controls how much weight to give the latest forecast vs the previous order. Low α = very smooth but slow to react. High α = more reactive but noisier.
""")

policy = st.sidebar.radio("Ordering Policy", [
    "Point Forecast (q0.5)",
    "Conservative Quantile",
    "Smoothed Orders",
])

if policy == "Conservative Quantile":
    q_level = st.sidebar.select_slider("Quantile Level", QUANTILES, value=0.25)
    qi      = QUANTILES.index(q_level)
    orders  = q_flat[:, qi]
    label   = f"TFT q{q_level}"

elif policy == "Smoothed Orders":
    alpha  = st.sidebar.slider("Smoothing Factor α", 0.05, 1.0, 0.3, 0.05,
                                help="Low α = smoother orders, slower to react. High α = more reactive, noisier.")
    orders = smooth(q_flat[:, 3], alpha)
    label  = f"TFT Smoothed (α={alpha})"

else:
    orders = q_flat[:, 3]
    label  = "TFT Point (q0.5)"

selected_ratio = bullwhip(actual_a, orders)
fill_rate      = np.mean(orders >= actual_a) * 100
reduction      = (naive_ratio - selected_ratio) / naive_ratio * 100

# --- Metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Naive Seasonal Ratio", f"{naive_ratio:.4f}",
          help="Bullwhip ratio when ordering based on last year's demand for the same month")
c2.metric(f"{label} Ratio", f"{selected_ratio:.4f}",
          delta=f"{reduction:+.1f}% vs Naive",
          delta_color="normal" if reduction > 0 else "inverse")
c3.metric("Fill Rate", f"{fill_rate:.1f}%",
          help="% of periods where the order quantity was enough to meet actual demand")
c4.metric("Demand Variance", f"{np.var(actual_a):,.0f}",
          help="Variance of actual customer demand — the baseline all policies are compared against")

st.divider()

# --- Policy comparison ---
st.subheader("Policy Comparison")
st.markdown("""
Each bar shows the bullwhip ratio for a different ordering strategy.
The dotted line at **1.0** is the ideal — orders exactly as volatile as demand.
Below 1.0 means orders are *smoother* than demand, which is the goal.
""")

all_policies = {
    "Naive Seasonal":        bullwhip(actual_a, naive_pred_a),
    "TFT Point (q0.5)":     bullwhip(actual_a, q_flat[:, 3]),
    "TFT q0.25":             bullwhip(actual_a, q_flat[:, 2]),
    "TFT Smoothed (α=0.3)": bullwhip(actual_a, smooth(q_flat[:, 3], 0.3)),
}

colors = ["#ff7f7f", "#ffbf7f", "#7fbf7f", "#7f9fbf"]
fig = go.Figure(go.Bar(
    x=list(all_policies.keys()),
    y=list(all_policies.values()),
    marker_color=colors,
    text=[f"{v:.3f}" for v in all_policies.values()],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Bullwhip Ratio: %{y:.4f}<extra></extra>",
))
fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
              annotation_text="No Bullwhip (1.0)", annotation_position="right")
fig.update_layout(yaxis_title="Bullwhip Ratio", height=380,
                  margin=dict(t=20, b=0), font=dict(size=14))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Alpha sensitivity ---
st.subheader("Smoothing Factor (α) Sensitivity")
st.markdown("""
This chart shows how the bullwhip ratio changes as you vary the smoothing factor α from 0.05 (very smooth)
to 1.0 (no smoothing — same as point forecast).

- **Lower α** → orders change slowly, bullwhip ratio drops, but the policy is slow to respond to real demand shifts.
- **Higher α** → orders track the forecast closely, bullwhip ratio rises toward the point forecast level.
- The **red dashed line** is the naive seasonal baseline. Any point below it means TFT smoothing outperforms naive ordering.
- The **grey dotted line** at 1.0 is the ideal (orders = demand volatility).
""")

alphas = np.arange(0.05, 1.01, 0.05)
ratios = [bullwhip(actual_a, smooth(q_flat[:, 3], a)) for a in alphas]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=alphas, y=ratios, mode="lines+markers",
    line=dict(color="#636EFA"), name="TFT Smoothed — Bullwhip Ratio",
    hovertemplate="α = %{x:.2f}<br>Bullwhip Ratio: %{y:.4f}<extra></extra>",
))
fig2.add_hline(y=naive_ratio, line_dash="dash", line_color="red",
               annotation_text=f"Naive Seasonal ({naive_ratio:.3f})", annotation_position="right")
fig2.add_hline(y=1.0, line_dash="dot", line_color="gray",
               annotation_text="No Bullwhip (1.0)", annotation_position="right")
fig2.update_layout(
    xaxis_title="Smoothing Factor α",
    yaxis_title="Bullwhip Ratio",
    legend=dict(orientation="h", y=-0.2),
    height=370, margin=dict(t=10, b=0), font=dict(size=14),
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Fill rate vs quantile ---
st.subheader("Fill Rate vs Bullwhip Ratio by Quantile")
st.markdown("""
This chart shows the **trade-off** between ordering conservatively (low bullwhip) and meeting demand (high fill rate).

- The **green line** (left axis) is the bullwhip ratio — lower is better for supply chain stability.
- The **red dashed line** (right axis) is the fill rate — the % of periods where the order was enough to cover actual demand. Higher is better for service level.
- Moving left (lower quantile) reduces the bullwhip ratio but also reduces the fill rate — you order less, so you risk stockouts more often.
- The **sweet spot** is the quantile where the bullwhip ratio is acceptably low without the fill rate dropping too far.
""")

fill_rates = [np.mean(np.maximum(q_flat[:, qi], 0) >= actual_a) * 100 for qi in range(7)]
bw_by_q    = [bullwhip(actual_a, q_flat[:, qi]) for qi in range(7)]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=QUANTILES, y=bw_by_q, name="Bullwhip Ratio",
    mode="lines+markers", line=dict(color="#7fbf7f"),
    hovertemplate="q%{x}<br>Bullwhip Ratio: %{y:.4f}<extra></extra>",
))
fig3.add_trace(go.Scatter(
    x=QUANTILES, y=fill_rates, name="Fill Rate (%)",
    mode="lines+markers", line=dict(color="#ff7f7f", dash="dash"),
    yaxis="y2",
    hovertemplate="q%{x}<br>Fill Rate: %{y:.1f}%<extra></extra>",
))
fig3.update_layout(
    xaxis_title="Order Quantile Level",
    yaxis=dict(title="Bullwhip Ratio", color="#7fbf7f", title_font=dict(color="#7fbf7f")),
    yaxis2=dict(title="Fill Rate (%)", overlaying="y", side="right",
                color="#ff7f7f", title_font=dict(color="#ff7f7f")),
    legend=dict(orientation="h", y=-0.2),
    height=400, margin=dict(t=10, b=0), font=dict(size=14),
)
st.plotly_chart(fig3, use_container_width=True)
