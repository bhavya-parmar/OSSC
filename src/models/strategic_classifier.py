"""
Minimax Strategic Linear Classifier (Algorithm 3).

This module implements the Full-Batch Subgradient Method for strategic classification
under unknown costs (Rosenfeld et al., ICML 2024).

Theoretical Formulation:
------------------------
Objective (Equation 4):
    min_{beta, b} max_{Sigma in C} [ 1/n sum_i max{0, 1 - y_i(beta^T x_i + b + u* ||beta||_{*, Sigma})} + lambda u* ||beta||_{*, Sigma} ]

Key Components:
1. Danskin's Theorem (Theorem 5.1):
   The subgradient of g(beta) := max_{Sigma in C} f(beta, Sigma) is given by the subgradient
   of f evaluated at any maximizing matrix Sigma* found by MAXLOSSCOST.
2. Dual Norm Regularization (Proposition 4.3):
   Adding lambda * u* * ||beta||_{*, Sigma} with lambda >= q(y = +1) guarantees that the
   regularized objective is strictly convex for all costs in C simultaneously.
3. Preconditioned Descent (Equations 56-62):
   Updates proceed along beta <- beta - eta_t * Sigma_min * g_beta, ensuring convergence
   in the transformed metric space at the theoretical minimax rate O(T^{-1/2}).
4. Best Iterate Selection:
   At termination, the algorithm returns the iterate beta^(t*) that achieved the minimum
   worst-case regularized objective over all rounds t in {1, ..., T}.
"""

import time
from typing import Optional, List
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..costs.uncertainty_set import CostUncertaintySet
from ..algorithms.max_loss_cost import max_loss_cost


class StrategicClassifier(BaseEstimator, ClassifierMixin):
    """
    Minimax Strategic Linear Classifier trained via Full-Batch Subgradient Descent.

    Parameters
    ----------
    uncertainty_set : CostUncertaintySet
        The continuous set of plausible cost matrices C.
    u_star : float, default=0.5
        Upper bound on applicant utility from obtaining a positive classification (+1).
    lam : float, optional
        Dual norm regularization strength lambda. If None, defaults to the empirical
        positive class frequency q(y=+1) which is the minimum value guaranteeing convexity
        (Proposition 4.3).
    n_iter : int, default=100
        Number of subgradient iterations T.
    learning_rate : float, default=0.3
        Base learning rate eta.
    fit_intercept : bool, default=True
        Whether to learn an unconstrained bias term b.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print loss progress per iteration.

    Attributes
    ----------
    coef_ : np.ndarray of shape (d,)
        Learned linear classifier weights beta.
    intercept_ : float
        Learned classifier bias b.
    loss_history_ : List[float]
        History of worst-case regularized risks at each iteration.
    best_loss_ : float
        The minimum worst-case risk achieved by the returned model.
    """

    def __init__(
        self,
        uncertainty_set: CostUncertaintySet,
        u_star: float = 0.5,
        lam: Optional[float] = None,
        n_iter: int = 100,
        learning_rate: float = 0.3,
        fit_intercept: bool = True,
        random_state: int = 42,
        verbose: bool = False
    ):
        self.uncertainty_set = uncertainty_set
        self.u_star = u_star
        self.lam = lam
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.loss_history_: List[float] = []
        self.best_loss_: float = float('inf')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StrategicClassifier':
        """
        Fits the strategic classifier using Algorithm 3 (Full-Batch Subgradient Method).

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Training feature matrix.
        y : np.ndarray of shape (n,)
            Target labels in {-1.0, +1.0}.

        Returns
        -------
        self : StrategicClassifier
            The fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        # Proposition 4.3: minimum lambda >= q(y = +1) guarantees convexity
        if self.lam is None:
            self.lam = float(np.mean(y == 1.0))

        # Initialize parameters beta^(1) = 0, b = 0
        beta = np.zeros(d, dtype=float)
        b = 0.0

        best_beta = beta.copy()
        best_b = b
        best_objective = float('inf')
        self.loss_history_ = []

        # Preconditioner matrix Sigma_min (eigenvalue bounds)
        sigma_min = self.uncertainty_set.sigma_min
        preconditioner = sigma_min ** 2

        start_time = time.time()
        for t in range(self.n_iter):
            # Step 1: Call MAXLOSSCOST subroutine to find worst-case Sigma_t and k_t
            Sigma_worst, sigma_worst_vec, k_val, _ = max_loss_cost(
                X=X, y=y, beta=beta, b=b,
                uncertainty_set=self.uncertainty_set,
                u_star=self.u_star,
                lam=self.lam,
                p=2
            )

            # Step 2: Compute current regularized strategic hinge loss
            scores = X @ beta + b
            margins = 1.0 - y * (scores + self.u_star * k_val)
            active = margins > 0
            loss_val = float(np.mean(np.maximum(0.0, margins)) + self.lam * self.u_star * k_val)
            self.loss_history_.append(loss_val)

            if loss_val < best_objective:
                best_objective = loss_val
                best_beta = beta.copy()
                best_b = b

            if self.verbose and (t % 20 == 0 or t == self.n_iter - 1):
                print(f"Iter [{t+1:3d}/{self.n_iter}]: Worst-Case Risk = {loss_val:.5f}, "
                      f"k_max = {k_val:.4f}, Best = {best_objective:.5f}")

            # Step 3: Compute subgradient g_t via Danskin's theorem
            # Dual norm gradient: grad_beta ||Sigma^{-1/2} beta||_2 = (Sigma^{-1} beta) / ||Sigma^{-1/2} beta||_2
            if k_val > 1e-12:
                grad_k = (beta / (sigma_worst_vec ** 2)) / k_val
            else:
                grad_k = np.zeros(d, dtype=float)

            y_active = y[active]
            X_active = X[active]
            sum_y_active = float(np.sum(y_active)) / n if np.sum(active) > 0 else 0.0

            # Loss gradient w.r.t beta: - 1/n sum_{i in active} y_i x_i - (sum_{i in active} y_i / n) u* grad_k
            if np.sum(active) > 0:
                grad_loss_beta = - (X_active.T @ y_active) / n
            else:
                grad_loss_beta = np.zeros(d, dtype=float)

            # Total subgradient for beta (Equations 58-62)
            g_beta = grad_loss_beta + (self.lam - sum_y_active) * self.u_star * grad_k
            g_b = - sum_y_active

            # Step 4: Preconditioned update with step decay eta_t = eta / sqrt(t + 1)
            eta_t = self.learning_rate / np.sqrt(t + 1)
            norm_precond = preconditioner / np.median(preconditioner)
            beta -= eta_t * (norm_precond * g_beta)
            if self.fit_intercept:
                b -= eta_t * g_b

        # Return best iterate t* achieving minimum regularized objective (Algorithm 3)
        self.coef_ = best_beta
        self.intercept_ = best_b
        self.best_loss_ = best_objective

        if self.verbose:
            elapsed = time.time() - start_time
            print(f"Training completed in {elapsed:.2f}s. Best worst-case regularized risk: {best_objective:.5f}")

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluates linear prediction score beta^T x + b."""
        if self.coef_ is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")
        return X @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions in {-1.0, +1.0}."""
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1.0, -1.0)

