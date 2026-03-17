"""Baseline models for demand forecasting."""

import numpy as np
import pandas as pd


class NaiveSeasonalBaseline:
    """Predicts using the same value from the same month last year (lag=12)."""

    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = seasonal_period

    def predict(self, series: pd.Series) -> pd.Series:
        """Return lagged values as predictions."""
        return series.shift(self.seasonal_period)

    def predict_df(self, df: pd.DataFrame, target_col: str = 'volume') -> pd.DataFrame:
        """Apply naive seasonal prediction per group."""
        df = df.copy()
        df['predicted'] = df.groupby(['agency', 'sku'])[target_col].shift(self.seasonal_period)
        return df.dropna(subset=['predicted'])
