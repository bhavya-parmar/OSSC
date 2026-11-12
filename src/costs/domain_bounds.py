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

