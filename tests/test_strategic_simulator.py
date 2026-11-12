"""
Test suite for StrategicSimulator.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.costs.uncertainty_set import CostUncertaintySet
from src.simulation.strategic_response import StrategicSimulator


def test_strategic_simulator_movement():
    d = 2
    bounds = np.array([[1.0, 2.0], [1.0, 2.0]])
    c_set = CostUncertaintySet(bounds)
    Sigma_star, sigma_star = c_set.sample_hidden_true_cost(random_state=42)

    simulator = StrategicSimulator(Sigma_star, sigma_star, utility_u=2.0)
    beta = np.array([1.0, 0.0])
    b = 0.0

    # Point 0: initial score is -0.5. With u=2.0, cost to boundary is 0.5 * sigma_star[0] < 2.0 -> moves
    # Point 1: initial score is -10.0. Cost is 10 * sigma_star[0] > 2.0 -> cannot afford to move
    # Point 2: initial score is +1.0. Already positive -> doesn't need to move
    X = np.array([
        [-0.5, 0.0],
        [-10.0, 0.0],
        [1.0, 0.0]
    ])

    X_manip, moved, costs = simulator.simulate(X, beta, b)

    assert moved[0] == True
    assert moved[1] == False
    assert moved[2] == False

    # Check that Point 0 crossed boundary strictly
    assert (X_manip[0] @ beta + b) >= 0.0
    # Check that Point 1 did not move
    assert np.allclose(X_manip[1], X[1])

