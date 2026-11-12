"""
Test suite for CostUncertaintySet and domain cost bounds.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.costs.uncertainty_set import CostUncertaintySet
from src.costs.domain_bounds import construct_domain_cost_bounds


def test_uncertainty_set_initialization():
    bounds = np.array([[1.0, 3.0], [2.0, 5.0], [0.5, 1.5]])
    c_set = CostUncertaintySet(bounds)
    assert c_set.d == 3
    assert np.allclose(c_set.sigma_min, [1.0, 2.0, 0.5])
    assert np.allclose(c_set.sigma_max, [3.0, 5.0, 1.5])

    # Error cases
    with pytest.raises(ValueError):
        CostUncertaintySet(np.array([[3.0, 1.0]]))  # min > max
    with pytest.raises(ValueError):
        CostUncertaintySet(np.array([[-1.0, 2.0]]))  # non-positive


def test_dual_norm_monotonicity():
    bounds = np.array([[1.0, 3.0], [2.0, 4.0]])
    c_set = CostUncertaintySet(bounds)
    beta = np.array([2.0, -1.5])

    k_min = c_set.min_dual_norm(beta)
    k_max = c_set.max_dual_norm(beta)
    assert k_min < k_max


def test_true_cost_sampling():
    bounds = np.array([[1.0, 3.0], [2.0, 6.0], [0.8, 2.4]])
    c_set = CostUncertaintySet(bounds)
    Sigma_star, sigma_star = c_set.sample_hidden_true_cost(random_state=99)

    assert Sigma_star.shape == (3, 3)
    assert c_set.is_valid_cost(sigma_star)
    assert np.all(sigma_star > c_set.sigma_min)
    assert np.all(sigma_star < c_set.sigma_max)

