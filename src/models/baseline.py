"""
Naive, non-strategic baseline classifier wrapper.
"""

from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    """
    Standard naive linear classifier trained without anticipating strategic gaming.

    Serves as the empirical benchmark to measure the vulnerability of standard
    supervised learning models when deployed against strategic applicants.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum solver iterations for logistic regression.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaselineClassifier':
        """Fits standard logistic regression on clean training data."""
        self.model.fit(X, y)
        self.coef_ = self.model.coef_[0]
        self.intercept_ = float(self.model.intercept_[0])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Returns signed continuous decision scores."""
        return self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions in {-1.0, +1.0}."""
        return self.model.predict(X)

