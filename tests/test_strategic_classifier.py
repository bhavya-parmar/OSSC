"""
Test suite for StrategicClassifier.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.costs.uncertainty_set import CostUncertaintySet
from src.models.strategic_classifier import StrategicClassifier


def test_strategic_classifier_scikit_learn_compliance():
    np.random.seed(42)
    n, d = 120, 5
    X = np.random.randn(n, d)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1.0, -1.0)
    bounds = np.array([[1.0, 3.0] for _ in range(d)])
    c_set = CostUncertaintySet(bounds)

    clf = StrategicClassifier(
        uncertainty_set=c_set,
        u_star=0.5,
        n_iter=25,
        learning_rate=0.25,
        random_state=42
    )

    clf.fit(X, y)

    # Check fitted attributes
    assert clf.coef_ is not None
    assert clf.coef_.shape == (d,)
    assert isinstance(clf.intercept_, float)
    assert len(clf.loss_history_) == 25
    assert clf.loss_history_[-1] < clf.loss_history_[0]

    # Predict checks
    preds = clf.predict(X)
    assert preds.shape == (n,)
    assert set(np.unique(preds)).issubset({-1.0, 1.0})

    scores = clf.decision_function(X)
    assert scores.shape == (n,)

