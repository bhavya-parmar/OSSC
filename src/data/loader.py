"""
Data Loading & Preprocessing Pipeline for Strategic Classification.

This module prepares real-world tabular data for strategic classification:
1. Missing value handling to preserve sample integrity.
2. One-hot encoding of categorical features.
3. Strict standardization (StandardScaler) of numerical features.
   * Crucial Theoretical Note: Feature scaling is strictly required in strategic
     classification because applicant manipulation cost is measured by a norm:
         c(x, x') = ||Sigma^{1/2}(x' - x)||
     Without standardization, dimensions with large natural scales (e.g. annual
     income in tens of thousands) would dominate or dwarf other dimensions (e.g.
     age in decades), distorting the eigenvalue bounds in the uncertainty set C.
4. Mapping target labels from {0, 1} to {-1, +1} to match the hinge loss convention:
     +1: Loan Approved (positive outcome desired by applicants)
     -1: Loan Denied (negative outcome that applicants seek to overturn)
5. Stratified train-test split for balanced empirical risk evaluation.
"""

import os
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_loan_data(
    csv_path: str,
    test_size: float = 0.25,
    random_state: int = 42,
    target_column: str = "loan_status"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Loads, cleans, encodes, scales, and partitions the loan dataset.

    Parameters
    ----------
    csv_path : str
        Path to the loan approval CSV dataset.
    test_size : float, default=0.25
        Fraction of samples reserved for the holdout test set.
    random_state : int, default=42
        Random seed for reproducible train-test splitting.
    target_column : str, default='loan_status'
        Name of the binary target variable column.

    Returns
    -------
    X_train : np.ndarray of shape (n_train, d)
        Preprocessed and scaled training feature matrix.
    X_test : np.ndarray of shape (n_test, d)
        Preprocessed and scaled holdout test feature matrix.
    y_train : np.ndarray of shape (n_train,)
        Training labels mapped to {-1.0, +1.0}.
    y_test : np.ndarray of shape (n_test,)
        Holdout test labels mapped to {-1.0, +1.0}.
    feature_names : List[str]
        Ordered list of column names for the d feature dimensions.
    scaler : StandardScaler
        The fitted scikit-learn StandardScaler used for numerical normalization.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Loan approval dataset not found at path: {csv_path}")

    # 1. Load raw data and clean missing values
    df = pd.read_csv(csv_path)
    df = df.dropna().copy()

    # 2. Distinguish categorical vs numerical columns
    cat_cols = [
        'person_gender', 'person_education', 'person_home_ownership',
        'loan_intent', 'previous_loan_defaults_on_file'
    ]
    num_cols = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score'
    ]

    # Keep only those present in the CSV
    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]

    # 3. Categorical one-hot encoding (drop_first=True prevents multicollinearity)
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 4. Extract target variable and map to {-1, +1}
    if target_column not in df_encoded.columns:
        raise ValueError(f"Target column '{target_column}' missing from dataset.")

    y_raw = df_encoded[target_column].values
    y = np.where(y_raw == 1, 1.0, -1.0)

    # Extract feature matrix dataframe
    X_df = df_encoded.drop(columns=[target_column])
    feature_names = X_df.columns.tolist()

    # 5. Stratified train/test split to preserve positive class prevalence
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6. Apply StandardScaler to numerical features
    scaler = StandardScaler()
    num_indices = [feature_names.index(col) for col in num_cols if col in feature_names]

    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)

    if len(num_indices) > 0:
        X_train[:, num_indices] = scaler.fit_transform(X_train[:, num_indices])
        X_test[:, num_indices] = scaler.transform(X_test[:, num_indices])

    return X_train, X_test, y_train, y_test, feature_names, scaler

