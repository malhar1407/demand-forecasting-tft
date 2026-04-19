import streamlit as st

st.set_page_config(
    page_title="TFT Demand Forecasting",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Interpretable Demand Forecasting")
st.markdown("#### Temporal Fusion Transformer for Supply Chain Optimization")
st.caption("M.Tech Project · Symbiosis Institute of Technology")

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "TFT (tuned)")
col2.metric("RMSE", "751.03", delta="-6.4% vs Naive", delta_color="normal")
col3.metric("MASE", "0.1431", delta="-1.6% vs Naive", delta_color="normal")
col4.metric("Bullwhip Reduction", "up to 49%", delta="vs Naive Seasonal", delta_color="normal")

st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("📈 Forecasts")
    st.write("6-month ahead demand predictions with 80% and 95% quantile intervals for any agency/SKU combination.")
    st.page_link("pages/1_Forecasts.py", label="Go to Forecasts →")

with c2:
    st.subheader("🔍 Interpretability")
    st.write("Encoder, decoder, and static variable importance scores plus per-sample attention weight visualisation.")
    st.page_link("pages/2_Interpretability.py", label="Go to Interpretability →")

with c3:
    st.subheader("🌊 Bullwhip Analysis")
    st.write("Compare point, conservative quantile, and smoothed ordering policies. Tune α and quantile level interactively.")
    st.page_link("pages/3_Bullwhip_Analysis.py", label="Go to Bullwhip Analysis →")

st.divider()
st.caption("Dataset: Stallion (Kaggle) · 21k records · 50 agency–SKU pairs · Monthly granularity")
