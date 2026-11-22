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

from typing import Tuple, Optional, Union, List
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

    def sample_multiple_interior_costs(
        self,
        n_samples: int = 10,
        random_state: Optional[int] = 42
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Samples N distinct random true cost matrices Sigma* strictly inside C.

        Parameters
        ----------
        n_samples : int, default=10
            Number of cost matrices to generate.
        random_state : int, optional
            Base seed for random sampling.

        Returns
        -------
        List of (Sigma_star, sigma_star) tuples.
        """
        costs = []
        base_seed = 42 if random_state is None else random_state
        for i in range(n_samples):
            cost_mat, cost_vec = self.sample_hidden_true_cost(random_state=base_seed + i * 17)
            costs.append((cost_mat, cost_vec))
        return costs

    def get_worst_case_cost(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the Adversarial / Worst-Case Cost Matrix Sigma_min.

        In this scenario, all features have the lowest possible effort cost:
        sigma_i = sigma_{i,min} for all i.
        This maximizes the dual norm k_max and maximizes the applicant's ability
        to manipulate their features across the decision boundary.
        """
        sigma_worst = self.sigma_min.copy()
        Sigma_worst = np.diag(sigma_worst ** 2)
        return Sigma_worst, sigma_worst

    def get_best_case_cost(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the Best-Case Cost Matrix Sigma_max (maximum manipulation friction).
        sigma_i = sigma_{i,max} for all i.
        """
        sigma_best = self.sigma_max.copy()
        Sigma_best = np.diag(sigma_best ** 2)
        return Sigma_best, sigma_best

    def get_skewed_cost(
        self,
        cheap_indices: Union[List[int], np.ndarray],
        expensive_indices: Optional[Union[List[int], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs a Skewed Cost Matrix where specific features are cheap (sigma_min)
        while others are expensive (sigma_max).

        For example, loan applicants can easily alter requested loan amount (low effort),
        while income and bureau records remain hard to alter (high effort).

        Parameters
        ----------
        cheap_indices : List[int] or np.ndarray
            Feature coordinate indices to set to sigma_{i,min}.
        expensive_indices : List[int] or np.ndarray, optional
            Feature coordinate indices to set to sigma_{i,max}. If None, all coordinates
            not in cheap_indices are set to sigma_{i,max}.

        Returns
        -------
        Sigma_skewed : np.ndarray of shape (d, d)
        sigma_skewed : np.ndarray of shape (d,)
        """
        sigma_skewed = np.zeros(self.d, dtype=float)
        cheap_set = set(cheap_indices)

        for i in range(self.d):
            if i in cheap_set:
                sigma_skewed[i] = self.sigma_min[i]
            else:
                sigma_skewed[i] = self.sigma_max[i]

        Sigma_skewed = np.diag(sigma_skewed ** 2)
        return Sigma_skewed, sigma_skewed

    def get_diameter(self) -> float:
        """
        Computes the theoretical diameter metric D(C) from Theorems 5.2 & 5.3 of Rosenfeld et al. (ICML 2024):
            D = max_{1 <= i <= d} ( 1 / sigma_{i,min}^2 - 1 / sigma_{i,max}^2 )

        Returns
        -------
        float
            The diameter of the cost uncertainty set C.
        """
        coord_diameters = (1.0 / (self.sigma_min ** 2)) - (1.0 / (self.sigma_max ** 2))
        return float(np.max(coord_diameters))

    @property
    def diameter(self) -> float:
        """Property shortcut for get_diameter()."""
        return self.get_diameter()

    def is_valid_cost(self, sigma_vec: np.ndarray, tol: float = 1e-6) -> bool:
        """Checks if a candidate cost vector lies within the uncertainty set C."""
        within_min = np.all(sigma_vec >= self.sigma_min - tol)
        within_max = np.all(sigma_vec <= self.sigma_max + tol)
        return bool(within_min and within_max)

