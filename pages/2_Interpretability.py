import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pytorch_forecasting import TimeSeriesDataSet
from src.models.loader import load_model_and_data, get_test_predictions

st.set_page_config(page_title="Interpretability", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 17px; }
    p, li, .stMarkdown { font-size: 16px !important; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Model Interpretability")
st.markdown("##### Understand what the TFT model learned and which variables drive its forecasts")

model, training, train, val, _ = load_model_and_data()
_, _, test_dataset = get_test_predictions()

@st.cache_resource
def get_interpretation(_model, _training, _train, _val):
    val_dataset = TimeSeriesDataSet.from_dataset(
        _training,
        pd.concat([_train, _val]).reset_index(drop=True),
        predict=True, stop_randomization=True,
    )
    loader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    raw = _model.predict(loader, mode="raw", return_x=True)
    return _model.interpret_output(raw.output, reduction="sum"), raw

interpretation, raw_preds = get_interpretation(model, training, train, val)

# --- Variable Importance ---
st.subheader("Variable Importance")

tab1, tab2, tab3 = st.tabs(["Encoder Variables", "Decoder Variables", "Static Variables"])

def importance_bar(series: pd.Series):
    series = series.sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=series.values, y=series.index, orientation="h",
                           marker_color="#636EFA"))
    fig.update_layout(xaxis_title="Importance Score", height=260,
                      margin=dict(l=0, r=20, t=10, b=0), font=dict(size=14))
    return fig

with tab1:
    st.markdown("""
**Encoder variables** are the historical inputs the model reads *before* making a forecast —
everything it knows about the past (e.g. past sales volume, past prices, past discounts).
Higher importance means the model relied on that variable more when building its understanding of the series.
""")
    enc = pd.Series(interpretation["encoder_variables"].numpy(), index=model.encoder_variables)
    st.plotly_chart(importance_bar(enc), use_container_width=True)

with tab2:
    st.markdown("""
**Decoder variables** are the future inputs the model is given *during* the forecast horizon —
things we already know about the future (e.g. planned prices, known promotions, calendar month).
Higher importance means the model leaned on that future information to adjust its predictions.
""")
    dec = pd.Series(interpretation["decoder_variables"].numpy(), index=model.decoder_variables)
    st.plotly_chart(importance_bar(dec), use_container_width=True)

with tab3:
    st.markdown("""
**Static variables** are fixed attributes that don't change over time — in this case the Agency and SKU identifiers.
The model uses these to learn a unique "profile" for each product–location combination,
allowing it to capture differences in baseline demand levels and seasonal patterns across groups.
""")
    stat = pd.Series(interpretation["static_variables"].numpy(), index=model.static_variables)
    st.plotly_chart(importance_bar(stat), use_container_width=True)

st.divider()

# --- Attention Weights ---
st.subheader("Attention Weights")

st.markdown("""
The attention plot shows **which past time steps the model focused on** when generating a forecast.

- The **blue line** is the observed (actual) demand over the encoder window (past ~24 months, shown as negative time indices) and into the forecast horizon.
- The **orange line** is the model's median forecast for the prediction horizon (positive time indices).
- The **orange shaded band** is the prediction interval (uncertainty range).
- The **grey line** is the attention weight (right y-axis) — it shows how much the model attended to each past time step. A spike at time index −12 means the model heavily relied on demand from 12 months ago, indicating it learned a seasonal pattern.

**How to interpret it:** Attention concentrated at regular intervals (e.g. −12, −24) signals the model learned seasonality. Attention spread evenly across the past means the model is averaging over a longer trend rather than reacting to a specific past event.
""")

sample_idx = st.slider("Sample index", 0, min(49, raw_preds.output.prediction.shape[0] - 1), 0,
                       help="Each index corresponds to one Agency–SKU prediction window in the validation set")

import matplotlib
matplotlib.rcParams.update({"font.size": 13})

fig_attn = model.plot_prediction(
    raw_preds.x, raw_preds.output, idx=sample_idx, add_loss_to_title=False
)
fig_attn.set_size_inches(7, 4)
for ax in fig_attn.axes:
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(4)
fig_attn.tight_layout()
col, _ = st.columns([2, 1])
col.pyplot(fig_attn)
