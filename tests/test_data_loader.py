"""
Test suite for data loading and preprocessing.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_and_preprocess_loan_data


def test_loan_data_loader():
    # Resolve CSV path
    possible_paths = [
        "data/loan_data.csv",
        "OSSC/data/loan_data.csv",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "loan_data.csv")
    ]
    csv_path = next((p for p in possible_paths if os.path.exists(p)), None)
    assert csv_path is not None, "loan_data.csv not found in test environment."

    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_loan_data(
        csv_path, test_size=0.25, random_state=42
    )

    # 1. Shape validations
    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert len(feature_names) == X_train.shape[1]
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]

    # 2. Target mapping validation {-1.0, +1.0}
    unique_labels = set(np.unique(y_train))
    assert unique_labels == {-1.0, 1.0}

    # 3. No NaNs or Infs
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isinf(X_train).any()
    assert not np.isinf(X_test).any()

