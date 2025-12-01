###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
##
## This script produces clustering for a group of points in 2D,
## using k-means for k = 2, 3, 4, 5. The input file is the csv
## file 'demo_data.csv' in folder 'data/'.
###

from __future__ import annotations

import os
import sys
from typing import List

# -----------------------------------------------------
# Ensure project root is on the Python path
# -----------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering

OUTPUT_DIR = "demo_output"


def main(args: List[str]) -> None:
    # -----------------------------------------------------
    # Validate command-line arguments
    # -----------------------------------------------------
    if len(args) != 1:
        print("Usage: python demo/cluster_plot.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------
    # Load CSV and validate numeric features
    # -----------------------------------------------------
    df = pd.read_csv(input_path)
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(numeric_cols) < 2:
        print("Error: The input CSV must contain at least two numeric columns.")
        sys.exit(1)

    # Use first two numeric columns for 2D clustering
    feature_cols = numeric_cols[:2]
    print(f"\n Using feature columns for clustering: {feature_cols}")

    # For file naming
    base = os.path.splitext(os.path.basename(input_path))[0]

    # -----------------------------------------------------
    # Run KMeans clustering for k = 2, 3, 4, 5
    # -----------------------------------------------------
    metrics_summary = []

    for k in (2, 3, 4, 5):
        print(f"\n=== Running k-means with k = {k} ===")

        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,
            standardise=True,
            output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv"),
            random_state=42,
            compute_elbow=False,
        )

        # Save cluster plot figure
        fig_cluster = result.get("fig_cluster")
        if fig_cluster is not None:
            plot_path = os.path.join(OUTPUT_DIR, f"{base}_k{k}.png")
            fig_cluster.savefig(plot_path, dpi=150)
            plt.close(fig_cluster)

        # Collect and print metrics
        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)

        print("Metrics:")
        for key, value in result.get("metrics", {}).items():
            print(f"  {key}: {value}")

    # -----------------------------------------------------
    # Save metrics summary
    # -----------------------------------------------------
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # -----------------------------------------------------
    # Plot silhouette score summary (if available)
    # -----------------------------------------------------
    # Standardise column naming
    if "silhouette" in metrics_df.columns:
        metrics_df.rename(columns={"silhouette": "silhouette_score"}, inplace=True)

    if "silhouette_score" in metrics_df.columns:
        fig = plt.figure()
        plt.bar(metrics_df["k"], metrics_df["silhouette_score"])
        plt.xlabel("k")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette score across k")
        stats_path = os.path.join(OUTPUT_DIR, f"{base}_silhouette.png")
        fig.savefig(stats_path, dpi=150)
        plt.close(fig)

    # -----------------------------------------------------
    # Completion message
    # -----------------------------------------------------
    print("\nDemo completed successfully.")
    print(f"All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])