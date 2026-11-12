"""
MAXLOSSCOST Subroutine (Algorithm 1 & Algorithm 3).

This module implements the exact inner-maximization algorithm from Rosenfeld et al. (ICML 2024):
    max_{Sigma in C} [ 1/n sum_i max{0, 1 - y_i(beta^T x_i + b + u* ||beta||_{*, Sigma})} + lambda u* ||beta||_{*, Sigma} ]

Mathematical Foundation:
------------------------
1. The empirical strategic hinge loss is piecewise linear with respect to the scalar dual norm:
       k = ||beta||_{*, Sigma} in [||beta||_min*, ||beta||_max*]
2. For each sample (x_i, y_i), the loss breakpoint occurs at:
       v_i = y_i * (1 - y_i * (beta^T x_i + b))
   - If y_i = +1: loss is max{0, v_i - u* k}. When k increases, loss decreases with slope -u*.
                  Reaches 0 at k = v_i / u*.
   - If y_i = -1: loss is max{0, u* k - v_i}. When k increases, loss increases with slope +u*.
                  Becomes positive at k = v_i / u*.
3. By sorting {v_i}_{i=1}^n in increasing order, we can sweep k across all linear segments in
   O(nd + n ln n) time, tracking the active sample counts:
       c_+1 = |{j' >= j : y_j' = +1}| (positive samples with positive loss)
       c_-1 = |{j' < j  : y_j' = -1}| (negative samples with positive loss)
   The slope of the regularized empirical risk on interval [k, k'] is:
       slope = lambda + (c_-1 - c_+1) / n
4. Once the maximizing dual norm scalar k_max is found, the algorithm analytically recovers
   the diagonal matrix Sigma in C that achieves ||beta||_{*, Sigma} = k_max via coordinate solve:
       sigma_hat_i = |beta_i| / ( k_max^p - ||Sigma_{i=0}^{-1/2} beta||_*^p )^{1/p}
"""

from typing import Tuple
import numpy as np
from ..costs.uncertainty_set import CostUncertaintySet


def max_loss_cost(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    b: float,
    uncertainty_set: CostUncertaintySet,
    u_star: float,
    lam: float,
    p: int = 2
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Executes the exact inner-maximization subroutine in O(nd + n ln n) time.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Training feature matrix.
    y : np.ndarray of shape (n,)
        Training labels {-1.0, +1.0}.
    beta : np.ndarray of shape (d,)
        Classifier weights.
    b : float
        Classifier intercept.
    uncertainty_set : CostUncertaintySet
        The cost uncertainty set C.
    u_star : float
        Upper bound on applicant utility.
    lam : float
        Dual norm regularization strength lambda.
    p : int, default=2
        Norm parameter (p=2 for Euclidean norm).

    Returns
    -------
    Sigma_worst : np.ndarray of shape (d, d)
        Worst-case diagonal cost matrix in C.
    sigma_worst_vec : np.ndarray of shape (d,)
        Worst-case cost factor vector.
    k_max : float
        The scalar dual norm that maximizes the objective.
    r_max : float
        The maximum worst-case regularized risk achieved.
    """
    n, d = X.shape
    sigma_min = uncertainty_set.sigma_min
    sigma_max = uncertainty_set.sigma_max

    k_min_val = uncertainty_set.min_dual_norm(beta, p=p)
    k_max_val = uncertainty_set.max_dual_norm(beta, p=p)

    # Degenerate case: zero weights or single-point uncertainty set
    if np.isclose(k_min_val, k_max_val) or k_max_val < 1e-12:
        return np.diag(sigma_min ** 2), sigma_min.copy(), k_min_val, 0.0

    scores = X @ beta + b
    # Step 1: Baseline scores v_i = y_i * (1 - y_i * (beta^T x_i + b))
    v = y * (1.0 - y * scores)

    # Step 2: Sort v_i in increasing order
    sort_idx = np.argsort(v)
    v_sorted = v[sort_idx]
    y_sorted = y[sort_idx]

    # Step 3: Initialize k <- ||beta||_min*, compute initial risk r
    k = k_min_val
    k_max = k
    initial_losses = np.maximum(0.0, 1.0 - y * (scores + u_star * k))
    r = float(np.mean(initial_losses) + lam * u_star * k)
    r_max = r

    # Step 4: Initialize index j <- min j s.t. v_j - u* k > 0
    cond = (v_sorted - u_star * k) > 0
    j = int(np.argmax(cond)) if np.any(cond) else n

    c_pos = int(np.sum(y_sorted[j:] == 1.0))
    c_neg = int(np.sum(y_sorted[:j] == -1.0))
    n_neg = int(np.sum(y == -1.0))

    # Step 5: Sweep k across breakpoints
    while k < k_max_val and j < n:
        k_prime = min(float(v_sorted[j] / u_star), k_max_val)
        if k_prime > k:
            slope = lam + (c_neg - c_pos) / n
            r += u_star * (k_prime - k) * slope
            k = k_prime
            if r > r_max:
                r_max = r
                k_max = k
        # Update active counts: c_{y_j} <- c_{y_j} - y_j
        if y_sorted[j] == 1.0:
            c_pos -= 1
        else:
            c_neg += 1
        j += 1

    # Check boundary condition at j == n
    if j == n and k < k_max_val:
        slope = lam + n_neg / n
        r_end = r + u_star * (k_max_val - k) * slope
        if r_end > r_max:
            r_max = r_end
            k_max = k_max_val

    # Step 6: Matrix Recovery (Find Sigma in C inducing ||beta||_{*, Sigma} = k_max)
    if k_max <= k_min_val + 1e-12:
        return np.diag(sigma_max ** 2), sigma_max.copy(), k_min_val, r_max
    if k_max >= k_max_val - 1e-12:
        return np.diag(sigma_min ** 2), sigma_min.copy(), k_max_val, r_max

    sigma_hat = sigma_max.copy().astype(float)
    for i in range(d):
        sigma_test = sigma_hat.copy()
        sigma_test[i] = sigma_min[i]
        norm_test = np.linalg.norm(beta / sigma_test, ord=p)

        if norm_test < k_max:
            sigma_hat[i] = sigma_min[i]
            continue

        # Analytical coordinate solve for dimension i
        other_dims_sum = np.sum((np.abs(np.delete(beta, i)) / np.delete(sigma_hat, i)) ** p)
        diff = (k_max ** p) - other_dims_sum
        if diff > 0 and np.abs(beta[i]) > 1e-12:
            val = np.abs(beta[i]) / (diff ** (1.0 / p))
            sigma_hat[i] = float(np.clip(val, sigma_min[i], sigma_max[i]))
        break

    Sigma_worst = np.diag(sigma_hat ** 2)
    return Sigma_worst, sigma_hat, k_max, r_max

