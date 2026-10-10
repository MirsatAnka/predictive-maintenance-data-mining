"""Multivariate time-series preprocessing and statistical feature extraction."""

from typing import List
import pandas as pd
from sklearn.preprocessing import RobustScaler


def load_sensor_telemetry(filepath: str) -> pd.DataFrame:
    """Load industrial telemetry dataset with automatic timestamp indexing."""
    df = pd.read_csv(filepath)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def extract_multivariate_statistics(
    df: pd.DataFrame, sensor_columns: List[str], window_sizes: List[int] = [5, 15, 60]
) -> pd.DataFrame:
    """Compute rolling statistical moments: mean, std, kurtosis, and crest factor."""
    result = df.copy()
    for col in sensor_columns:
        for w in window_sizes:
            roll = result[col].rolling(window=w, min_periods=1)
            result[f"{col}_mean_w{w}"] = roll.mean()
            result[f"{col}_std_w{w}"] = roll.std().fillna(0)
            result[f"{col}_peak_to_peak_w{w}"] = roll.max() - roll.min()
            # Crest factor surrogate: max / rms
            rms = (result[col].pow(2).rolling(window=w, min_periods=1).mean()).pow(0.5)
            result[f"{col}_crest_w{w}"] = (roll.max() / (rms + 1e-6)).fillna(0)
    return result


def scale_telemetry_features(
    df: pd.DataFrame, feature_columns: List[str]
) -> pd.DataFrame:
    """Robust scaling resilient to equipment transient spikes/outliers."""
    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(df[feature_columns])
    scaled_df = pd.DataFrame(scaled_values, columns=feature_columns, index=df.index)
    return scaled_df
