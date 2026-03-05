"""Data preprocessing for Stallion dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def load_stallion_data(data_path: str = "data/raw/train_OwBvO8W") -> pd.DataFrame:
    """Load and merge all Stallion dataset files."""
    data_path = Path(data_path)
    
    # Load main datasets
    historical_volume = pd.read_csv(data_path / 'historical_volume.csv')
    price_promo = pd.read_csv(data_path / 'price_sales_promotion.csv')
    demographics = pd.read_csv(data_path / 'demographics.csv')
    
    # Convert YearMonth to datetime
    historical_volume['date'] = pd.to_datetime(historical_volume['YearMonth'], format='%Y%m')
    price_promo['date'] = pd.to_datetime(price_promo['YearMonth'], format='%Y%m')
    
    # Merge datasets
    data = historical_volume.merge(
        price_promo, 
        on=['date', 'Agency', 'SKU'], 
        how='left'
    )
    data = data.merge(demographics, on='Agency', how='left')
    
    return data


def preprocess_for_tft(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for TFT model."""
    
    # Rename columns to lowercase for consistency
    data = data.rename(columns={
        'Agency': 'agency',
        'SKU': 'sku',
        'Volume': 'volume',
        'Price': 'price_regular',
        'Promotions': 'discount'
    })
    
    # Sort by agency, sku, and date
    data = data.sort_values(['agency', 'sku', 'date']).reset_index(drop=True)
    
    # Create time index per group
    data['time_idx'] = data.groupby(['agency', 'sku']).cumcount()
    
    # Add month feature
    data['month'] = data['date'].dt.month.astype(str)
    
    # Create log volume (target transformation)
    data['log_volume'] = np.log1p(data['volume'])
    
    # Fill missing values
    data['price_regular'] = data['price_regular'].fillna(data['price_regular'].median())
    data['discount'] = data['discount'].fillna(0)
    
    # Select required columns
    columns = [
        'agency', 'sku', 'date', 'time_idx', 'month',
        'volume', 'log_volume', 'price_regular', 'discount'
    ]
    
    return data[columns]


def create_train_val_test_split(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets based on time."""
    
    # Get max time index
    max_time_idx = data['time_idx'].max()
    
    # Calculate split points
    train_end = int(max_time_idx * train_ratio)
    val_end = int(max_time_idx * (train_ratio + val_ratio))
    
    # Split data
    train = data[data['time_idx'] <= train_end].copy()
    val = data[(data['time_idx'] > train_end) & (data['time_idx'] <= val_end)].copy()
    test = data[data['time_idx'] > val_end].copy()
    
    print(f"Train: {len(train)} samples (time_idx: 0-{train_end})")
    print(f"Val: {len(val)} samples (time_idx: {train_end+1}-{val_end})")
    print(f"Test: {len(test)} samples (time_idx: {val_end+1}-{max_time_idx})")
    
    return train, val, test


def save_processed_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_path: str = "data/processed"
):
    """Save processed datasets."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(output_path / 'train.csv', index=False)
    val.to_csv(output_path / 'val.csv', index=False)
    test.to_csv(output_path / 'test.csv', index=False)
    
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    print("Loading Stallion dataset...")
    data = load_stallion_data()
    
    print(f"\nRaw data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    print("\nPreprocessing data...")
    data = preprocess_for_tft(data)
    
    print(f"\nProcessed data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"\nSample data:")
    print(data.head())
    
    print("\nSplitting data...")
    train, val, test = create_train_val_test_split(data)
    
    print("\nSaving processed data...")
    save_processed_data(train, val, test)
    
    print("\n✓ Preprocessing complete!")
