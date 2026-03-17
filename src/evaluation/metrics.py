"""Evaluation metrics for demand forecasting."""

import numpy as np
import pandas as pd


def mase(actual: np.ndarray, predicted: np.ndarray, seasonal_period: int = 1) -> float:
    """Mean Absolute Scaled Error."""
    mae = np.mean(np.abs(actual - predicted))
    naive_mae = np.mean(np.abs(actual[seasonal_period:] - actual[:-seasonal_period]))
    return mae / naive_mae


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8))


def quantile_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    """Quantile (pinball) loss."""
    errors = actual - predicted
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray, name: str = "Model") -> dict:
    """Compute all metrics for a forecast."""
    metrics = {
        "model": name,
        "mase": mase(actual, predicted, seasonal_period=12),
        "smape": smape(actual, predicted),
        "mae": np.mean(np.abs(actual - predicted)),
        "rmse": np.sqrt(np.mean((actual - predicted) ** 2)),
    }
    print(f"\n{name} Metrics:")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k.upper()}: {v:.4f}")
    return metrics
