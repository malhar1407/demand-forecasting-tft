"""Shared model and data loader for Streamlit app."""

import streamlit as st
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer

MAX_ENCODER_LENGTH = 24
MAX_PREDICTION_LENGTH = 6
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
MODEL_PATH = "models/tft_tuned_best-v2.ckpt"


@st.cache_resource
def load_model_and_data():
    train = pd.read_csv("data/processed/train.csv", parse_dates=["date"])
    val   = pd.read_csv("data/processed/val.csv",   parse_dates=["date"])
    test  = pd.read_csv("data/processed/test.csv",  parse_dates=["date"])

    for df in [train, val, test]:
        df["month"] = df["month"].astype(str)

    training = TimeSeriesDataSet(
        train,
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["agency", "sku"],
        time_varying_known_categoricals=["month"],
        time_varying_known_reals=["time_idx", "price_regular", "discount"],
        time_varying_unknown_reals=["volume", "log_volume"],
        target_normalizer=GroupNormalizer(groups=["agency", "sku"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    model = TemporalFusionTransformer.load_from_checkpoint(
        MODEL_PATH, map_location=torch.device("cpu")
    )
    model.eval()

    return model, training, train, val, test


@st.cache_resource
def get_test_predictions():
    model, training, train, val, test = load_model_and_data()

    test_dataset = TimeSeriesDataSet.from_dataset(
        training,
        pd.concat([train, val, test]).reset_index(drop=True),
        predict=True,
        stop_randomization=True,
    )
    loader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    raw_preds = model.predict(loader, mode="raw", return_y=True)
    actual = raw_preds.y[0].numpy()                       # (samples, horizon)
    quantile_preds = raw_preds.output.prediction.numpy()  # (samples, horizon, 7)

    return actual, quantile_preds, test_dataset
