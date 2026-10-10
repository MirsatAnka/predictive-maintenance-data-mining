"""CLI Pipeline for predictive maintenance anomaly mining."""

import argparse
import pandas as pd
from src.preprocessing import extract_multivariate_statistics, scale_telemetry_features
from src.mining import (
    mine_operational_clusters,
    detect_density_anomalies,
    score_multivariate_anomalies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive Maintenance Telemetry Miner")
    parser.add_argument("--data-path", type=str, required=True, help="Path to raw telemetry CSV")
    parser.add_argument("--contamination", type=float, default=0.03, help="Expected anomaly ratio")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    sensor_cols = [c for c in df.select_dtypes(include=["float64", "int64"]).columns if c != "label"]
    
    # 1. Feature extraction
    fe_df = extract_multivariate_statistics(df, sensor_cols, window_sizes=[5, 15])
    feature_cols = [c for c in fe_df.columns if c not in ["timestamp", "label"]]
    
    # 2. Scaling
    scaled_df = scale_telemetry_features(fe_df, feature_cols)
    X = scaled_df.values

    # 3. Mining
    cluster_labels, _ = mine_operational_clusters(X, n_clusters=3)
    dbscan_labels = detect_density_anomalies(X, eps=1.2, min_samples=8)
    iso_flags, scores = score_multivariate_anomalies(X, contamination=args.contamination)

    fe_df["op_cluster"] = cluster_labels
    fe_df["dbscan_anomaly"] = (dbscan_labels == -1).astype(int)
    fe_df["iso_anomaly"] = (iso_flags == -1).astype(int)
    fe_df["anomaly_score"] = scores

    anomaly_count = (fe_df["iso_anomaly"] == 1).sum()
    print(f"Mining complete. Analyzed {len(fe_df)} telemetry frames.")
    print(f"Identified {anomaly_count} anomalous sensor operating frames ({anomaly_count/len(fe_df)*100:.2f}%).")


if __name__ == "__main__":
    main()
