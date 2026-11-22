"""
Domain-specific cost bounds construction for loan approval features.
"""

from typing import List
import numpy as np
from .uncertainty_set import CostUncertaintySet


def construct_domain_cost_bounds(feature_names: List[str]) -> CostUncertaintySet:
    """
    Constructs domain-grounded cost uncertainty bounds for loan features.

    Domain Rationale:
    -----------------
    1. Immutables / Verified Records (High Effort: sigma in [8.0, 16.0]):
       - person_age, cb_person_cred_hist_length, person_gender_male,
         previous_loan_defaults_on_file_Yes.
       - Biological traits and bureau records cannot be legitimately falsified or
         manipulated by loan applicants without severe legal penalty or impossibility.

    2. Low-Effort / Stated Application Fields (Low Effort: sigma in [0.6, 1.5]):
       - loan_amnt, loan_intent_*.
       - Modifying the requested loan amount or selecting a different stated purpose
         on a web form involves near-zero friction for an applicant trying to game approval.

    3. Moderate Financial Effort (Moderate Effort: sigma in [1.8, 3.8]):
       - person_income, credit_score.
       - Increasing reported income (e.g. side gigs) or boosting credit scores (paying down
         credit card balances) requires real, non-trivial financial effort over months.

    4. Moderate Structural Effort (sigma in [2.5, 5.0]):
       - person_education_*, person_home_ownership_*.
       - Completing degree programs or changing home ownership status (e.g. rent vs own)
         takes substantial time and capital.

    Parameters
    ----------
    feature_names : List[str]
        List of all d feature column names.

    Returns
    -------
    CostUncertaintySet
        The initialized continuous uncertainty set C.
    """
    bounds = []
    for name in feature_names:
        low = name.lower()
        if any(k in low for k in ['age', 'cred_hist', 'gender', 'default']):
            # High effort / immutable
            bounds.append([8.0, 16.0])
        elif any(k in low for k in ['loan_amnt', 'intent']):
            # Low effort / highly manipulable
            bounds.append([0.6, 1.5])
        elif any(k in low for k in ['income', 'credit_score']):
            # Moderate financial effort
            bounds.append([1.8, 3.8])
        elif any(k in low for k in ['education', 'home_ownership']):
            # Moderate structural effort
            bounds.append([2.5, 5.0])
        else:
            # Default bounds for other features
            bounds.append([1.5, 4.0])

    return CostUncertaintySet(np.array(bounds))


def construct_scaled_uncertainty_set(
    feature_names: List[str],
    uncertainty_pct: float = 0.5,
    min_sigma_floor: float = 0.1
) -> CostUncertaintySet:
    """
    Constructs an uncertainty set C where interval width is scaled by uncertainty_pct (e.g., +/- 10%, +/- 50%, +/- 90%).

    This directly tests the core theoretical bounds from Theorems 5.2 & 5.3 of Rosenfeld et al. (ICML 2024),
    where sub-optimality scales with the diameter of C:
        D(C) = max_i ( 1 / sigma_{i,min}^2 - 1 / sigma_{i,max}^2 )

    Parameters
    ----------
    feature_names : List[str]
        List of all d feature column names.
    uncertainty_pct : float, default=0.5
        Uncertainty radius around nominal center:
        - 0.10 (+/- 10%): High confidence / narrow uncertainty.
        - 0.50 (+/- 50%): Moderate confidence / baseline uncertainty.
        - 0.90 (+/- 90%): Very low confidence / wide uncertainty.
    min_sigma_floor : float, default=0.1
        Absolute lower floor for sigma_{i,min} to prevent division by zero or negative costs.

    Returns
    -------
    CostUncertaintySet
        The parameterized uncertainty set C with corresponding diameter D.
    """
    base_set = construct_domain_cost_bounds(feature_names)
    scaled_bounds = []

    for i in range(base_set.d):
        s_min = base_set.sigma_min[i]
        s_max = base_set.sigma_max[i]
        center = 0.5 * (s_min + s_max)

        # Apply percentage width
        new_min = max(min_sigma_floor, center * (1.0 - uncertainty_pct))
        new_max = center * (1.0 + uncertainty_pct)

        if new_min >= new_max:
            new_max = new_min + 0.1

        scaled_bounds.append([new_min, new_max])

    return CostUncertaintySet(np.array(scaled_bounds))


