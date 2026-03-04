import streamlit as st

st.set_page_config(
    page_title="TFT Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Interpretable Demand Forecasting with TFT")
st.markdown("### Temporal Fusion Transformer for Supply Chain Optimization")

st.info("👈 Select a page from the sidebar to explore forecasts, interpretability, and bullwhip analysis")

st.markdown("""
## Project Overview
This application demonstrates interpretable demand forecasting using Temporal Fusion Transformers (TFT).

### Key Features:
- **Multi-horizon Forecasting**: Predict demand 6 months ahead
- **Interpretability**: Attention weights and variable importance
- **Bullwhip Effect Analysis**: Quantify supply chain variance amplification

### Pages:
1. **Forecasts**: View predictions vs actuals
2. **Interpretability**: Explore attention weights and feature importance
3. **Bullwhip Analysis**: Simulate supply chain dynamics
""")
