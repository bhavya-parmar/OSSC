"""
Cost Uncertainty Set & Dual Norm Formulations.

This module formalizes the unknown cost model from Section 2 of Rosenfeld et al. (ICML 2024):
- The applicant's cost to manipulate features from x to x' is parameterized by a positive definite
  matrix Sigma:
      c(x, x') = phi(||Sigma^{1/2}(x' - x)||)
- When Sigma is diagonal, Sigma = diag(sigma_1^2, ..., sigma_d^2), so:
      ||Sigma^{1/2}(x' - x)||_2 = sqrt( sum_{i=1}^d sigma_i^2 (x'_i - x_i)^2 )
- The continuous uncertainty set C is defined as coordinate-wise eigenvalue bounds:
      C = { Sigma = diag(sigma_1^2, ..., sigma_d^2) : sigma_i in [sigma_{i,min}, sigma_{i,max}] }
- The maximal strategic shift an agent can induce on their classifier score (beta^T x)
  is bounded by the dual norm (Lemma 2.3):
      Delta score_max = u* ||beta||_{*, Sigma} = u* ||Sigma^{-1/2} beta||_*
                      = u* sqrt( sum_{i=1}^d (beta_i / sigma_i)^2 )
"""

from typing import Tuple, Optional, Union
import numpy as np


class CostUncertaintySet:
    """
    Continuous uncertainty set C of positive definite diagonal cost matrices.

    Parameters
    ----------
    bounds : np.ndarray of shape (d, 2)
        Array of lower and upper cost bounds [[sigma_1_min, sigma_1_max], ...].
        Each sigma_i represents the effort multiplier along feature dimension i.

    Attributes
    ----------
    d : int
        Feature dimensionality.
    sigma_min : np.ndarray of shape (d,)
        Lower cost bounds (cheapest effort, induces maximum dual norm k_max).
    sigma_max : np.ndarray of shape (d,)
        Upper cost bounds (hardest effort, induces minimum dual norm k_min).
    """

    def __init__(self, bounds: np.ndarray):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(f"bounds must be of shape (d, 2), got {self.bounds.shape}")
        if np.any(self.bounds[:, 0] <= 0):
            raise ValueError("All lower cost bounds sigma_{i,min} must be strictly positive.")
        if np.any(self.bounds[:, 0] > self.bounds[:, 1]):
            raise ValueError("sigma_{i,min} cannot exceed sigma_{i,max}.")

        self.d = self.bounds.shape[0]
        self.sigma_min = self.bounds[:, 0]
        self.sigma_max = self.bounds[:, 1]

    def dual_norm(self, beta: np.ndarray, sigma_vec: np.ndarray, p: int = 2) -> float:
        """
        Computes the Sigma-transformed dual norm ||beta||_{*, Sigma} = ||Sigma^{-1/2} beta||_* (Definition 2.1).

        For diagonal Sigma with diagonal elements sigma_i^2:
            ||Sigma^{-1/2} beta||_2 = sqrt( sum_{i=1}^d (beta_i / sigma_i)^2 )

        Parameters
        ----------
        beta : np.ndarray of shape (d,)
            Classifier weight vector.
        sigma_vec : np.ndarray of shape (d,)
            Vector of cost factors sigma_i where Sigma = diag(sigma_vec^2).
        p : int, default=2
            Base norm order. Dual norm order is q where 1/p + 1/q = 1.

        Returns
        -------
        float
            Dual norm value.
        """
        if p == 2:
            return float(np.sqrt(np.sum((beta / sigma_vec) ** 2)))
        return float(np.linalg.norm(beta / sigma_vec, ord=p))

    def min_dual_norm(self, beta: np.ndarray, p: int = 2) -> float:
        """
        Computes ||beta||_min* := min_{Sigma in C} ||beta||_{*, Sigma}.
        Achieved when each sigma_i is maximized (sigma_i = sigma_{i,max}).
        """
        return self.dual_norm(beta, self.sigma_max, p=p)

    def max_dual_norm(self, beta: np.ndarray, p: int = 2) -> float:
        """
        Computes ||beta||_max* := max_{Sigma in C} ||beta||_{*, Sigma}.
        Achieved when each sigma_i is minimized (sigma_i = sigma_{i,min}).
        """
        return self.dual_norm(beta, self.sigma_min, p=p)

    def sample_hidden_true_cost(self, random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples a hidden 'True Cost Matrix' Sigma* strictly inside the uncertainty set C.

        Neither the naive baseline nor the strategic training algorithm has access
        to this matrix. It is strictly reserved for test-time rational agent simulation.

        Parameters
        ----------
        random_state : int, optional
            Seed for random sampling.

        Returns
        -------
        Sigma_star : np.ndarray of shape (d, d)
            Diagonal true cost matrix Sigma* = diag(sigma_star^2).
        sigma_star : np.ndarray of shape (d,)
            Diagonal cost factor vector.
        """
        rng = np.random.RandomState(random_state)
        alpha = rng.uniform(0.2, 0.8, size=self.d)
        sigma_star = self.sigma_min + alpha * (self.sigma_max - self.sigma_min)
        Sigma_star = np.diag(sigma_star ** 2)
        return Sigma_star, sigma_star

    def is_valid_cost(self, sigma_vec: np.ndarray, tol: float = 1e-6) -> bool:
        """Checks if a candidate cost vector lies within the uncertainty set C."""
        within_min = np.all(sigma_vec >= self.sigma_min - tol)
        within_max = np.all(sigma_vec <= self.sigma_max + tol)
        return bool(within_min and within_max)

