###
## cluster_maker - Tesing the preprocessing.py file
## Georgie Paterson - University of Bath
## December 2025
###

import unittest
import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):
    # ------------------------------------------------------------------
    # TEST 1: select_features must raise KeyError when a requested column
    # does not exist. This catches a real and important data-validation
    # bug: silently ignoring missing columns would cause incorrect or
    # incomplete clustering later in the pipeline.
    # ------------------------------------------------------------------
    def test_select_features_raises_for_missing_column(self):
        df = pd.DataFrame({
            "height": [1.7, 1.6, 1.8],
            "weight": [70, 65, 72],
        })
        # Asking for a non-existent column must raise KeyError
        with self.assertRaises(KeyError):
            select_features(df, ["height", "age"])  # "age" is missing


    # ------------------------------------------------------------------
    # TEST 2: select_features must raise TypeError for non-numeric
    # columns. This protects the clustering algorithms, which require
    # numeric inputs. If select_features fails to enforce this, the
    # entire clustering pipeline will break or produce invalid results.
    # ------------------------------------------------------------------
    def test_select_features_rejects_non_numeric_columns(self):
        df = pd.DataFrame({
            "height": [1.7, 1.8, 1.6],
            "name": ["Alice", "Bob", "Cara"],  # non-numeric
        })
        # Selecting a non-numeric column must raise TypeError
        with self.assertRaises(TypeError):
            select_features(df, ["height", "name"])


    # ------------------------------------------------------------------
    # TEST 3: standardise_features must output data with zero mean and
    # unit variance. If standardisation is wrong, clustering distances
    # become distorted and cluster assignments become unreliable.
    # ------------------------------------------------------------------
    def test_standardise_features_zero_mean_unit_variance(self):
        X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ])
        X_scaled = standardise_features(X)

        # Mean should be (approximately) zero
        self.assertTrue(np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-7))

        # Standard deviation should be (approximately) one
        self.assertTrue(np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-7))

if __name__ == "__main__":
    unittest.main()
