# EXPLANATION.md

## 1. What was wrong with the original demo script

The original `cluster_plot.py` script contained a logic error in the call to `run_clustering()`:

\`\`\`python
k = min(k, 3)
\`\`\`

Although the script looped over `k = 2, 3, 4, 5`, this line forced clustering to only ever run with **k ≤ 3**, meaning:

- The intended runs for **k = 4** and **k = 5** never actually occurred  
- The CSV and plot outputs labelled “k4” and “k5” incorrectly contained the results for **k = 3**  
- The metrics summary claimed four separate clustering runs, but only two unique runs were produced  

This caused the demo to behave incorrectly even though it did not crash.

A second problem appeared when running:

```
python demo/cluster_plot.py data/demo_data.csv
```

which produced:

```
ModuleNotFoundError: No module named 'cluster_maker'
```

Python does not automatically include the project root in `sys.path`, so the package could not be imported.

---

## 2. How the script was fixed

### (a) Correcting the k-value logic

The incorrect line:

\`\`\`python
k = min(k, 3)
\`\`\`

was replaced with:

\`\`\`python
k = k
\`\`\`

Now the script correctly performs clustering for **k = 2, 3, 4, and 5**.

### (b) Fixing the module import error

The following lines were added near the top of the script:

\`\`\`python
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
\`\`\`

This ensures Python can correctly import the `cluster_maker` package when the script is executed directly.

### Additional improvements

- Updated the usage message  
- Printed which numeric columns were selected  
- Normalised silhouette metric naming (`silhouette` → `silhouette_score`)  

---

## 3. What the corrected demo script now does

The corrected script:

1. Loads the input CSV  
2. Identifies numeric columns and selects the first two  
3. Runs k-means clustering for **k = 2, 3, 4, 5**  
4. Saves:
   - Clustered CSVs  
   - Scatter plots with centroids  
   - A metrics summary CSV  
   - A silhouette bar chart (if available)  
5. Prints inertia and silhouette scores  
6. Stores all outputs in `demo_output/`

The script now behaves exactly as intended.

---

## 4. Overview of the `cluster_maker` package
`cluster_maker` is a Python package built to demonstrate the key steps of a clustering workflow. It provides simple tools for creating data, preprocessing it, running clustering algorithms, evaluating the results and visualising the outcome.


## Key Features
- Build seed DataFrames that define cluster centres and generate noisy simulated datasets.
- Run complete clustering pipelines (preprocessing → clustering → evaluation → plotting).
- Produce quick 2D cluster plots and elbow plots.
- Compute basic metrics to assess clustering quality.
- Provide simple tools to select, validate, and prepare features for clustering.

---

## Module-by-Module Functionality Overview

Below is a summary of what each module does and the purpose of its key functions.

---

## 1. `dataframe_builder.py` – Synthetic Data Construction

This module creates artificial clustered datasets.

### Main functions

- **`define_dataframe_structure(column_specs)`**  
  Builds a DataFrame where each row represents a cluster and each column contains a list of *representative values (“reps”)* for that cluster.  
  This defines the cluster centres.

- **`simulate_data(seed_df, n_points, cluster_std, random_state)`**  
  Generates noisy data points around the cluster centres using Gaussian noise.  
  Returns a DataFrame of simulated samples with a `true_cluster` column indicating which cluster generated each point.

---

## 2. `preprocessing.py` – Data Selection, Scaling

Handles feature preparation before clustering.

### Main functions

- **`select_features(data, feature_cols)`**  
  Selects specified columns from the DataFrame and ensures they are numeric.

- **`standardise_features(X)`**  
  Standardises each feature to zero mean and unit variance using `StandardScaler`.

- **`pca_transform(X, n_components)`**  
  Performs PCA using SVD, returning the data projected onto the first principal components.

---

## 3. `algorithms.py` – Clustering Algorithms

Implements clustering logic, including a manual K-Means algorithm.

### Main functions

- **`init_centroids(X, k, random_state)`**  
  Selects `k` random data points as initial centroids.

- **`assign_clusters(X, centroids)`**  
  Assigns each point to the nearest centroid using Euclidean distance.

- **`update_centroids(X, labels, k)`**  
  Recalculates centroid positions based on assigned points, reinitialising empty clusters.

- **`kmeans(X, k, max_iter, tol, random_state)`**  
  Full manual K-Means loop: initialise → assign → update → repeat until convergence.

- **`sklearn_kmeans(X, k, random_state)`**  
  Wrapper around scikit-learn’s `KMeans`, returning labels and centroids.

---

## 4. `data_analyser.py` – Data Summaries and Inspection

Provides tools to understand datasets before and after clustering.

### Main functions

- **`calculate_descriptive_statistics(data)`**  
  Returns count, mean, std, quartiles, etc. for numeric columns.

- **`calculate_correlation(data)`**  
  Computes a correlation matrix between numeric features.

- **`column_summary(df)`**  
  Returns a table of column-by-column statistics, including missing values.  
  Numeric and non-numeric columns are handled appropriately.

---

## 5. `evaluation.py` – Clustering Metrics

Measures clustering performance and supports choosing the number of clusters.

### Main functions

- **`compute_inertia(X, labels, centroids)`**  
  Calculates within-cluster sum of squared distances (compactness measure).

- **`silhouette_score_sklearn(X, labels)`**  
  Computes silhouette score using scikit-learn.

- **`elbow_curve(X, k_values, random_state, use_sklearn)`**  
  Returns inertia values for multiple `k` values, used to draw an elbow plot.

---

## 6. `plotting_clustered.py` – Visualisation

Creates graphical outputs to help interpret clustering results.

### Main functions

- **`plot_clusters_2d(X, labels, centroids, title)`**  
  Creates a 2D scatter plot of clustered data with optional centroid markers.

- **`plot_elbow(k_values, inertias, title)`**  
  Plots inertia vs. number of clusters for the elbow method.

---

## 7. `data_exporter.py` – Output and File Saving

Handles exporting datasets and summaries.

### Main functions

- **`export_to_csv(data, filename)`**  
  Saves a DataFrame to a CSV file.

- **`export_formatted(data, file)`**  
  Writes the DataFrame as a formatted text table.

- **`export_summary(summary_df, csv_path, txt_path)`**  
  Saves summary statistics to CSV and a readable text summary.

---

## 8. `interface.py` – High-Level Workflow Controller

The central entry point of the package.

- **`run_clustering(...)`**  
  Orchestrates the entire clustering pipeline:  
  1. Loads the input CSV  
  2. Selects and preprocesses features  
  3. Optionally standardises or applies PCA  
  4. Runs the chosen clustering algorithm  
  5. Computes metrics (inertia, silhouette)  
  6. Creates plots (cluster and elbow)  
  7. Optionally exports labelled data  
  8. Returns all results in a dictionary  

This function integrates all other modules into one coherent workflow.

---