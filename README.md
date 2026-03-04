# Interpretable Demand Forecasting using Temporal Fusion Transformers

Master's level project on demand forecasting with interpretability using TFT to reduce bullwhip effect in supply chains.

## Project Overview
- **Model**: Temporal Fusion Transformer (TFT)
- **Dataset**: Stallion (Kaggle) - 21k sales records
- **Key Features**: Multi-horizon forecasting, attention-based interpretability, bullwhip effect analysis
- **Deployment**: Streamlit app on Streamlit Community Cloud

## Project Structure
```
demand-forecasting-tft/
├── data/                    # Data storage
│   ├── raw/                 # Original Stallion dataset
│   ├── processed/           # Preprocessed TimeSeriesDataSet
│   └── external/            # Additional data
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_tft_analysis.ipynb
├── src/                     # Source code
│   ├── data/                # Data preprocessing
│   ├── models/              # Model implementations
│   ├── evaluation/          # Metrics and evaluation
│   └── visualization/       # Plotting utilities
├── configs/                 # Configuration files
├── scripts/                 # Training/evaluation scripts
├── tests/                   # Unit tests
├── models/                  # Saved model checkpoints
├── results/                 # Metrics, plots, outputs
├── pages/                   # Streamlit multi-page app
├── app.py                   # Streamlit app entry point
└── requirements.txt
```

## Setup

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Training on Kaggle
1. Push code to GitHub
2. Clone in Kaggle notebook
3. Run training scripts with GPU
4. Download trained models

## Usage

### Data Preprocessing
```bash
python scripts/preprocess.py
```

### Training
```bash
python scripts/train.py --config configs/model_config.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --model_path models/tft_best.ckpt
```

### Run Streamlit App
```bash
streamlit run app.py
```

## Results
- TFT vs Baselines comparison
- Variable importance analysis
- Attention weight visualization
- Bullwhip effect reduction metrics

## Author
Malhar Shinde  
M.Tech in Artificial Intelligence and Machine Learning  
Symbiosis Institute of Technology

## License
MIT
