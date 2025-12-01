###
## cluster_maker: demo for simulating clusters
## Georgie Paterson - University of Bath
## December 2025
###

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure cluster_maker is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from cluster_maker import run_clustering


OUTPUT_DIR = "demo_output_simulated"


def main() -> None:
    print("\n===============================================================")
    print("        SIMULATED DATA CLUSTERING ANALYSIS (DEMO)              ")
    print("===============================================================\n")

    # ----------------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------------
    data_path = os.path.join(ROOT, "data", "simulated_data.csv")
    print(f"Loading dataset from: {data_path}")

    if not os.path.exists(data_path):
        print("ERROR: Could not find simulated_data.csv. Aborting.")
        return

    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    # Expect at least 2 numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) < 2:
        print("ERROR: Dataset must contain at least 2 numeric columns.")
        return

    feature_cols = numeric_cols[:2]
    print(f"Using feature columns: {feature_cols}\n"
          "These columns define the 2D space in which clustering occurs.\n")

    # Make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------------
    # 1. Visualise raw data structure
    # ----------------------------------------------------------------------
    print("STEP 1: Visualising the raw data distribution...")
    print("This plot helps reveal natural groupings or shapes before clustering.\n")

    plt.figure(figsize=(6, 5))
    plt.scatter(df[feature_cols[0]], df[feature_cols[1]], s=20, alpha=0.7)
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.title("Raw data distribution (simulated_data.csv)")
    raw_plot_path = os.path.join(OUTPUT_DIR, "raw_scatter.png")
    plt.savefig(raw_plot_path, dpi=150)
    plt.close()

    print(f"Saved raw data plot → {raw_plot_path}\n")

    # ----------------------------------------------------------------------
    # 2. Try different k values and evaluate clustering quality
    # ----------------------------------------------------------------------
    print("STEP 2: Running clustering for a range of k values (2–6)...")
    print("This lets us compare how well different numbers of clusters fit the data.\n")

    k_values = [2, 3, 4, 5, 6]
    metrics_list = []

    for k in k_values:
        print(f"--- Now clustering with k = {k} ---")
        print("Standardising features and running k-means...")

        result = run_clustering(
            input_path=data_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,
            standardise=True,
            compute_elbow=False,
            output_path=os.path.join(OUTPUT_DIR, f"simulated_clustered_k{k}.csv"),
            random_state=42,
        )

        labels = result["labels"]

        # Save cluster plot
        fig_cluster = result["fig_cluster"]
        cluster_plot_path = os.path.join(OUTPUT_DIR, f"clusters_k{k}.png")
        fig_cluster.savefig(cluster_plot_path, dpi=150)
        plt.close(fig_cluster)

        inertia = result["metrics"].get("inertia", None)
        silhouette = result["metrics"].get("silhouette", None)

        print(f"  → inertia = {inertia:.3f}  (lower = tighter clusters)")
        sil_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
        print(f"  → silhouette = {sil_str}  (higher = clearer separation)")
        print(f"  Saved k={k} cluster plot → {cluster_plot_path}\n")

        metrics_list.append({"k": k, "inertia": inertia, "silhouette": silhouette})

    # Save metrics summary
    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics summary CSV → {metrics_csv_path}\n")

    # ----------------------------------------------------------------------
    # 3. Plot silhouette vs k to identify plausible number of clusters
    # ----------------------------------------------------------------------
    print("STEP 3: Analysing silhouette scores across k...")
    print("Silhouette score measures how well-separated the clusters are.")
    print("Higher values usually indicate a more natural number of clusters.\n")

    plt.figure(figsize=(6, 5))
    plt.plot(metrics_df["k"], metrics_df["silhouette"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score vs k (simulated data)")
    silhouette_plot_path = os.path.join(OUTPUT_DIR, "silhouette_vs_k.png")
    plt.savefig(silhouette_plot_path, dpi=150)
    plt.close()

    print(f"Saved silhouette vs k plot → {silhouette_plot_path}\n")

    # ----------------------------------------------------------------------
    # 4. Choose k using silhouette peak
    # ----------------------------------------------------------------------
    best_row = metrics_df.loc[metrics_df["silhouette"].idxmax()]
    best_k = int(best_row["k"])
    best_score = best_row["silhouette"]

    print("STEP 4: Selecting a plausible clustering...")
    print(f"The highest silhouette score is {best_score:.3f}, achieved when k = {best_k}.")
    print("This suggests that the dataset has approximately this many natural clusters.\n")

    # Run clustering again for best k
    print(f"Running final clustering using k = {best_k}...")
    final_result = run_clustering(
        input_path=data_path,
        feature_cols=feature_cols,
        algorithm="kmeans",
        k=best_k,
        standardise=True,
        compute_elbow=False,
        output_path=os.path.join(OUTPUT_DIR, f"simulated_final_clustered_k{best_k}.csv"),
        random_state=42,
    )

    fig_final = final_result["fig_cluster"]
    final_plot_path = os.path.join(OUTPUT_DIR, f"final_clusters_k{best_k}.png")
    fig_final.savefig(final_plot_path, dpi=150)
    plt.close(fig_final)

    print(f"Saved FINAL cluster plot (k={best_k}) → {final_plot_path}\n")

    # ----------------------------------------------------------------------
    # Completion message
    # ----------------------------------------------------------------------
    print("===============================================================")
    print("                     ANALYSIS COMPLETE")
    print("---------------------------------------------------------------")
    print(f"All outputs have been saved to: {OUTPUT_DIR}")
    print("You may inspect the plots to verify that the chosen k makes sense.")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
