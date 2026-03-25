"""Hyperparameter tuning for TFT using Optuna."""

import sys
sys.path.append('..')

import pandas as pd
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from src.models.tft_model import create_datasets

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, train: pd.DataFrame, val: pd.DataFrame) -> float:
    """Optuna objective function - returns validation loss."""

    # Suggest hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
    dropout = trial.suggest_float("dropout", 0.1, 0.3, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 3e-2, log=True)

    # Create datasets
    training, validation = create_datasets(train, val)
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    # Create model
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size // 4,
        loss=QuantileLoss(),
        optimizer="ranger",
        reduce_on_plateau_patience=3,
    )

    # Train with early stopping
    trainer = pl.Trainer(
        max_epochs=15,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    # Load data
    train = pd.read_csv("data/processed/train.csv", parse_dates=["date"])
    val = pd.read_csv("data/processed/val.csv", parse_dates=["date"])

    for df in [train, val]:
        df["month"] = df["month"].astype(str)

    # Run Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train, val), n_trials=10, show_progress_bar=True)

    # Results
    print("\n=== Best Hyperparameters ===")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest val loss: {study.best_value:.4f}")

    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv("results/optuna_results.csv", index=False)
    print("\nResults saved to results/optuna_results.csv")
