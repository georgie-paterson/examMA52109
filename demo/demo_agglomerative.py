###
## cluster_maker: demo for agglomerative clustering
## Georgie Paterson - University of Bath
## December 2025
##

"""
demo_agglomerative.py
---------------------

Demonstration script for hierarchical agglomerative clustering using the
cluster_maker package.

This script:
1) Loads difficult_dataset.csv
2) Uses run_clustering() with algorithm="agglomerative"
3) Tests several linkage methods
4) Produces clear visualisations and printed explanations
5) Helps the user decide which linkage gives the most meaningful clustering

Outputs are saved in demo_output_agglomerative/.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the cluster_maker package is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from cluster_maker import run_clustering
from cluster_maker.preprocessing import select_features


OUTPUT_DIR = "demo_output_agglomerative"


def main() -> None:
    print("\n======================================================")
    print("     AGGLOMERATIVE CLUSTERING DEMO (HIERARCHICAL)")
    print("======================================================\n")

    print("This demonstration applies hierarchical agglomerative clustering")
    print("to a challenging dataset containing four concentric ring-shaped")
    print("groups. The aim is to illustrate how different linkage strategies")
    print("behave on non-convex cluster shapes. The script will:")
    print("  • load the dataset;")
    print("  • try several linkage methods ('ward', 'complete', 'average');")
    print("  • compute clustering quality metrics (inertia, silhouette);")
    print("  • produce plots showing the resulting partitions;")
    print("  • identify which linkage performs best by silhouette score.\n")
    print("Because the clusters are ring-shaped, hierarchical methods cannot")
    print("perfectly recover the true structure, but the comparison remains")
    print("informative and demonstrates the algorithm's behaviour.\n")


    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    data_path = os.path.join(ROOT, "data", "difficult_dataset.csv")
    print(f"Loading dataset from: {data_path}\n")

    if not os.path.exists(data_path):
        print("ERROR: difficult_dataset.csv not found.")
        return

    df = pd.read_csv(data_path)

    # Use the first 2 numeric columns for 2D visualisation
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        print("ERROR: dataset must contain at least 2 numeric columns.")
        return

    feature_cols = numeric_cols[:2]
    print(f"Using features: {feature_cols}")
    print("These two features define the 2D space for plotting.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot raw data
    # ------------------------------------------------------------------
    print("STEP 1: Plotting the raw data distribution...")
    X_raw = select_features(df, feature_cols).to_numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(X_raw[:, 0], X_raw[:, 1], s=20, alpha=0.7)
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.title("Raw data distribution — difficult_dataset.csv")
    raw_out = os.path.join(OUTPUT_DIR, "raw_data.png")
    plt.savefig(raw_out, dpi=150)
    plt.close()
    print(f"Saved raw data plot → {raw_out}\n")

    # ------------------------------------------------------------------
    # Try multiple linkage strategies
    # ------------------------------------------------------------------
    linkages = ["ward", "complete", "average"]
    print("STEP 2: Trying multiple agglomerative linkage strategies:")
    print(" - 'ward'     → compact clusters")
    print(" - 'complete' → emphasises furthest distances")
    print(" - 'average'  → balanced merging method\n")

    results = {}   # will store metrics for each method

    for method in linkages:
        print(f"--- Running agglomerative clustering (linkage = '{method}') ---")

        result = run_clustering(
            input_path=data_path,
            feature_cols=feature_cols,
            algorithm="agglomerative",
            k=4,
            linkage=method,
            standardise=True,
            compute_elbow=False,
            random_state=42,
            output_path=os.path.join(OUTPUT_DIR, f"difficult_clustered_{method}.csv"),
        )

        labels = result["labels"]
        inertia = result["metrics"]["inertia"]
        silhouette = result["metrics"]["silhouette"]

        print(f"  Inertia:    {inertia:.3f}")
        print(f"  Silhouette: {silhouette:.3f}\n")

        # Save cluster plot
        fig = result["fig_cluster"]
        outpath = os.path.join(OUTPUT_DIR, f"clusters_{method}.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)

        print(f"Saved cluster plot → {outpath}\n")

        # Store metrics
        results[method] = {
            "inertia": inertia,
            "silhouette": silhouette,
        }

    # ------------------------------------------------------------------
    # Select best linkage by silhouette score
    # ------------------------------------------------------------------
    print("STEP 3: Selecting the best linkage method...\n")

    best = max(results, key=lambda m: results[m]["silhouette"])
    best_score = results[best]["silhouette"]

    print(f"Best linkage method based on silhouette score:")
    print(f" → '{best}'  (silhouette = {best_score:.3f})\n")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("======================================================")
    print("      INTERPRETATION — WHICH LINKAGE WORKS BEST?")
    print("======================================================\n")

    print("Agglomerative clustering is distance-based and therefore struggles")
    print("to recover non-convex shapes such as the concentric rings present")
    print("in this dataset. This is not an implementation error but a known")
    print("limitation of hierarchical clustering methods.\n")

    print("Among the tested options, the silhouette metric selected:")
    print(f" • Best method: '{best}'\n")

    print("Complete and average linkage typically behave more sensibly than")
    print("ward linkage on irregular or curved clusters, though none can")
    print("perfectly recover ring-shaped groups.\n")

    print("Demo complete.\n")


if __name__ == "__main__":
    main()
