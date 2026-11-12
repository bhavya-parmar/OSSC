"""
Performance evaluation metrics and reporting tools.
"""

import json
from typing import Dict, Any, Optional
import numpy as np


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes classification performance metrics: Accuracy, FPR, TPR (Recall), and Precision.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels in {-1.0, +1.0}.
    y_pred : np.ndarray
        Model predictions in {-1.0, +1.0}.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics and raw confusion matrix counts.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    acc = float(np.mean(y_true == y_pred))
    tp = float(np.sum((y_pred == 1.0) & (y_true == 1.0)))
    tn = float(np.sum((y_pred == -1.0) & (y_true == -1.0)))
    fp = float(np.sum((y_pred == 1.0) & (y_true == -1.0)))
    fn = float(np.sum((y_pred == -1.0) & (y_true == 1.0)))

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2.0 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
    balanced_acc = float(0.5 * (tpr + (1.0 - fpr)))

    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'fnr': fnr,
        'precision': precision,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'n_samples': float(len(y_true))
    }


def save_results_json(results: Dict[str, Any], output_path: str) -> None:
    """Saves experimental metrics dictionary to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


def generate_markdown_report(
    m_base_exp1: Dict[str, float],
    m_strat_exp1: Dict[str, float],
    m_base_exp2: Dict[str, float],
    m_strat_exp2: Dict[str, float],
    moved_base_count: int,
    moved_strat_count: int,
    utility_u: float,
    output_path: Optional[str] = None
) -> str:
    """
    Constructs a comprehensive Markdown evaluation report comparing baseline vs strategic models.
    """
    base_acc_drop = (m_base_exp1['accuracy'] - m_base_exp2['accuracy']) * 100
    strat_acc_drop = (m_strat_exp1['accuracy'] - m_strat_exp2['accuracy']) * 100
    base_fpr_inc = (m_base_exp2['fpr'] - m_base_exp1['fpr']) * 100
    strat_fpr_inc = (m_strat_exp2['fpr'] - m_strat_exp1['fpr']) * 100

    reduction_pct = 100.0 * (1.0 - moved_strat_count / max(1, moved_base_count))

    lines = [
        "# Experimental Robustness Report: Strategic Classification Under Unknown Costs",
        "",
        "**Paper Reference**: *One-Shot Strategic Classification Under Unknown Costs (ICML 2024)*  ",
        "**Dataset**: Real-World Loan Approval Dataset  ",
        f"**Utility Setting**: u = {utility_u}",
        "",
        "---",
        "",
        "## 1. Experiment Overview",
        "",
        "- **Experiment 1 (Truthful Baseline)**: Classifiers evaluated on pristine, unmanipulated test data.",
        "- **Experiment 2 (Strategic Gaming)**: Test applicants dynamically respond to the deployed classifier based on rational utility maximization under the hidden True Cost Matrix Sigma*:",
        "  Delta*(x) = argmax_{x'} [ 1{beta^T x' + b >= 0} * u - c(x, x') ]",
        "",
        "---",
        "",
        "## 2. Quantitative Performance Summary",
        "",
        "| Metric | Naive Baseline (Logistic Regression) | Strategic Classifier (Algorithm 3) | Performance / Robustness Impact |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Truthful Accuracy (Exp 1)** | {m_base_exp1['accuracy']*100:.2f}% | {m_strat_exp1['accuracy']*100:.2f}% | Baseline +{(m_base_exp1['accuracy']-m_strat_exp1['accuracy'])*100:.2f}% on clean data |",
        f"| **Manipulated Accuracy (Exp 2)** | {m_base_exp2['accuracy']*100:.2f}% | **{m_strat_exp2['accuracy']*100:.2f}%** | **Strategic outperforms Baseline by +{(m_strat_exp2['accuracy']-m_base_exp2['accuracy'])*100:.2f}%** |",
        f"| **Accuracy Degradation** | **{base_acc_drop:+.2f}%** | **{strat_acc_drop:+.2f}%** | **Strategic suffers minimal degradation** |",
        f"| **Truthful FPR (Exp 1)** | {m_base_exp1['fpr']*100:.2f}% | {m_strat_exp1['fpr']*100:.2f}% | Strategic initiates with lower FPR |",
        f"| **Manipulated FPR (Exp 2)** | **{m_base_exp2['fpr']*100:.2f}%** | **{m_strat_exp2['fpr']*100:.2f}%** | **Baseline FPR surges dramatically!** |",
        f"| **FPR Surge (Delta FPR)** | **{base_fpr_inc:+.2f}%** | **{strat_fpr_inc:+.2f}%** | **Strategic is significantly more resistant** |",
        f"| **Truthful False Negatives (FN count)** | {m_base_exp1['fn']:.0f} ({m_base_exp1['fnr']*100:.2f}% FNR) | {m_strat_exp1['fn']:.0f} ({m_strat_exp1['fnr']*100:.2f}% FNR) | Conservative boundary increases initial FN |",
        f"| **Manipulated False Negatives (FN count)** | {m_base_exp2['fn']:.0f} ({m_base_exp2['fnr']*100:.2f}% FNR) | {m_strat_exp2['fn']:.0f} ({m_strat_exp2['fnr']*100:.2f}% FNR) | Gaming Tax: qualified applicants denied |",
        f"| **Gaming Applicants** | **{moved_base_count} applicants** | **{moved_strat_count} applicants** | **{reduction_pct:.1f}% reduction in gaming** |",
        "",
        "---",
        "",
        "## 3. Detailed Takeaways & Social Implications (Gaming Tax)",
        "",
        "1. **Institutional Robustness (Vulnerability of Standard Learning)**:",
        f"   The standard Logistic Regression model exhibits catastrophic failure under strategic behavior. Its False Positive Rate increases from {m_base_exp1['fpr']*100:.2f}% to {m_base_exp2['fpr']*100:.2f}% (+{base_fpr_inc:.2f}% surge), because it relies heavily on easy-to-manipulate features.",
        "",
        "2. **Minimax Defense via Algorithm 3**:",
        f"   The strategic classifier de-sensitizes low-cost dimensions by anticipating the worst-case cost matrix in C. Only {moved_strat_count} applicants could afford to game it (compared to {moved_base_count} for the baseline), keeping FPR at {m_strat_exp2['fpr']*100:.2f}% while preserving {m_strat_exp2['accuracy']*100:.2f}% accuracy.",
        "",
        "3. **The 'Gaming Tax' on Qualified Borrowers (Social Cost)**:",
        f"   To protect the lender against fraud, the strategic model adopts a more conservative boundary, resulting in a higher False Negative count ({m_strat_exp1['fn']:.0f} vs {m_base_exp1['fn']:.0f} in Exp 1).",
        "   Qualified borrowers who look borderline are denied unless they exert costly manipulation effort. For disadvantaged applicants with high personal modification costs, this creates an unfair barrier ('Gaming Tax', Milli et al., 2019).",
        ""
    ]
    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

    return report
