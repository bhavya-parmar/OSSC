"""
Strategic Misreporting Simulation (Rational Agent Response).

This module simulates the rational strategic behavior of test-set applicants
facing an unknown true cost matrix Sigma* (Equation 1 & Lemma 2.3).

Mathematical Formulation:
------------------------
An applicant with true features x solves:
    x(beta, b) = argmax_{x'} [ 1{beta^T x' + b >= 0} * u - c(x, x') ]
where the true cost function is:
    c(x, x') = ||Sigma^{* 1/2}(x' - x)||_2

Optimization Solution:
----------------------
1. If beta^T x + b >= 0:
   The applicant is already approved (+1); cost = 0; no movement: x' = x.
2. If beta^T x + b < 0:
   The applicant receives a negative prediction (-1). To overturn the decision,
   they solve:
       min_{Delta} ||Sigma^{* 1/2} Delta||_2  subject to  beta^T (x + Delta) + b >= 0
   By Cauchy-Schwarz, the minimal cost to reach the decision boundary is:
       cost^*(x) = - (beta^T x + b) / ||beta||_{*, Sigma^*}
3. Rational Movement Decision:
   The applicant evaluates their net utility:
       Net Utility = u - cost^*(x)
   - If Net Utility > 0 (cost^*(x) < u):
     The applicant dynamically manipulates their features via optimal displacement:
         Delta^*(x) = ( - (beta^T x + b) + eps ) / ( ||beta||_{*, Sigma^*}^2 ) * (Sigma^{* -1} beta)
         x' = x + Delta^*(x)
     This pushes their score strictly across the decision boundary: beta^T x' + b = eps > 0.
   - If Net Utility <= 0 (cost^*(x) >= u):
     The cost to reach the boundary exceeds the loan utility u; the applicant stays at x' = x.
"""

from typing import Tuple
import numpy as np


class StrategicSimulator:
    """
    Simulates rational applicant feature gaming under a hidden true cost matrix Sigma*.

    Parameters
    ----------
    Sigma_star : np.ndarray of shape (d, d)
        The true positive definite diagonal cost matrix Sigma* = diag(sigma_star^2).
    sigma_star_vec : np.ndarray of shape (d,)
        The true cost factor vector sigma_star.
    utility_u : float, default=0.5
        The payoff u received by an applicant upon loan approval (+1).
    """

    def __init__(self, Sigma_star: np.ndarray, sigma_star_vec: np.ndarray, utility_u: float = 0.5):
        self.Sigma_star = np.asarray(Sigma_star, dtype=float)
        self.sigma_star = np.asarray(sigma_star_vec, dtype=float)
        self.utility_u = float(utility_u)
        self.inv_sigma_sq = 1.0 / (self.sigma_star ** 2)

    def simulate(
        self,
        X: np.ndarray,
        beta: np.ndarray,
        b: float,
        eps: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies utility-maximizing manipulation to test applicant features.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Original truthful test feature matrix.
        beta : np.ndarray of shape (d,)
            Deployed classifier weights.
        b : float
            Deployed classifier intercept.
        eps : float, default=1e-4
            Small buffer beyond the boundary to guarantee strict positive classification.

        Returns
        -------
        X_manipulated : np.ndarray of shape (n, d)
            The manipulated feature matrix after strategic adaptation.
        moved_mask : np.ndarray of shape (n,), dtype=bool
            Boolean array indicating which applicants manipulated their features.
        costs : np.ndarray of shape (n,), dtype=float
            The strategic cost incurred by each applicant (0.0 for non-movers).
        """
        scores = X @ beta + b
        dual_norm_star = float(np.sqrt(np.sum((beta / self.sigma_star) ** 2)))

        n = X.shape[0]
        X_manip = X.copy()
        moved_mask = np.zeros(n, dtype=bool)
        costs = np.zeros(n, dtype=float)

        # If beta is effectively zero, gaming is meaningless
        if dual_norm_star < 1e-12:
            return X_manip, moved_mask, costs

        # Identify rejected applicants
        negative_mask = (scores < 0)

        # Compute minimum cost to reach decision boundary
        req_cost = np.where(negative_mask, -scores / dual_norm_star, 0.0)

        # Rational agent condition: Net Utility = u - cost > 0
        can_move = negative_mask & (req_cost < self.utility_u)

        moved_indices = np.where(can_move)[0]
        moved_mask[moved_indices] = True
        costs[moved_indices] = req_cost[moved_indices]

        # Optimal displacement vector: Delta^* = scale * (Sigma^{* -1} beta)
        disp_dir = self.inv_sigma_sq * beta
        disp_scale = (-scores[moved_indices] + eps) / (dual_norm_star ** 2)
        X_manip[moved_indices] += disp_scale[:, None] * disp_dir[None, :]

        return X_manip, moved_mask, costs

