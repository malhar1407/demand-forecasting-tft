"""TFT model definition and dataset configuration."""

import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


MAX_ENCODER_LENGTH = 24
MAX_PREDICTION_LENGTH = 6


def create_datasets(train: pd.DataFrame, val: pd.DataFrame):
    """Create TFT-compatible TimeSeriesDataSet for train and validation."""

    # Ensure month is string type
    train = train.copy()
    val = val.copy()
    train['month'] = train['month'].astype(str)
    val['month'] = val['month'].astype(str)

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

    validation = TimeSeriesDataSet.from_dataset(
        training,
        pd.concat([train, val]).reset_index(drop=True),
        predict=True,
        stop_randomization=True,
    )

    return training, validation


def create_tft_model(training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Create TFT model from dataset."""
    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=64,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        optimizer="ranger",
        reduce_on_plateau_patience=4,
    )
