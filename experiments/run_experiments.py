"""
Experiment Runner: Comparative Evaluation of Strategic vs Baseline Classifiers.

Executes:
1. Data loading and feature preprocessing
2. Uncertainty set definition and hidden true cost sampling
3. Model training for Baseline Logistic Regression and Minimax Strategic Classifier
4. Experiment 1: Truthful evaluation (No misreporting)
5. Experiment 2: Strategic manipulation simulation (Net utility > 0)
6. Automated generation of:
   - Formatted terminal report
   - Markdown summary report (reports/evaluation_report.md)
   - JSON metrics artifact (reports/experiment_results.json)
   - Publication-quality visualization charts (reports/figures/*.png)
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Ensure src package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data.loader import load_and_preprocess_loan_data
from src.costs.domain_bounds import construct_domain_cost_bounds
from src.models.baseline import BaselineClassifier
from src.models.strategic_classifier import StrategicClassifier
from src.simulation.strategic_response import StrategicSimulator
from src.evaluation.metrics import (
    calculate_metrics,
    save_results_json,
    generate_markdown_report
)


def plot_evaluation_figures(
    m_base_exp1: Dict[str, float],
    m_strat_exp1: Dict[str, float],
    m_base_exp2: Dict[str, float],
    m_strat_exp2: Dict[str, float],
    loss_history: list,
    output_dir: str
) -> None:
    """Generates visual comparison figures for the reports folder."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. False Positive Rate (FPR) Surge Bar Chart
    plt.figure(figsize=(7, 5))
    models = ['Baseline (Naive)', 'Strategic (Algorithm 3)']
    x = np.arange(len(models))
    width = 0.35

    truthful_fpr = [m_base_exp1['fpr'] * 100, m_strat_exp1['fpr'] * 100]
    manip_fpr = [m_base_exp2['fpr'] * 100, m_strat_exp2['fpr'] * 100]

    plt.bar(x - width / 2, truthful_fpr, width, label='Exp 1: Truthful', color='#3498db', alpha=0.85)
    plt.bar(x + width / 2, manip_fpr, width, label='Exp 2: Manipulated', color='#e74c3c', alpha=0.85)

    plt.ylabel('False Positive Rate (%)', fontsize=12)
    plt.title('Vulnerability vs Robustness: False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    plt.xticks(x, models, fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(models)):
        plt.text(x[i] - width / 2, truthful_fpr[i] + 0.3, f"{truthful_fpr[i]:.2f}%", ha='center', fontsize=10)
        plt.text(x[i] + width / 2, manip_fpr[i] + 0.3, f"{manip_fpr[i]:.2f}%", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fpr_comparison.png"), dpi=200)
    plt.close()

    # 2. Accuracy Comparison Bar Chart
    plt.figure(figsize=(7, 5))
    truthful_acc = [m_base_exp1['accuracy'] * 100, m_strat_exp1['accuracy'] * 100]
    manip_acc = [m_base_exp2['accuracy'] * 100, m_strat_exp2['accuracy'] * 100]

    plt.bar(x - width / 2, truthful_acc, width, label='Exp 1: Truthful', color='#2ecc71', alpha=0.85)
    plt.bar(x + width / 2, manip_acc, width, label='Exp 2: Manipulated', color='#f39c12', alpha=0.85)

    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy Degradation Under Strategic Gaming', fontsize=14, fontweight='bold')
    plt.xticks(x, models, fontsize=11)
    plt.ylim([70, 95])
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(models)):
        plt.text(x[i] - width / 2, truthful_acc[i] + 0.3, f"{truthful_acc[i]:.2f}%", ha='center', fontsize=10)
        plt.text(x[i] + width / 2, manip_acc[i] + 0.3, f"{manip_acc[i]:.2f}%", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=200)
    plt.close()

    # 3. Social Implications: False Negative Rate (FNR) & Gaming Tax Bar Chart
    plt.figure(figsize=(7, 5))
    truthful_fnr = [m_base_exp1['fnr'] * 100, m_strat_exp1['fnr'] * 100]
    manip_fnr = [m_base_exp2['fnr'] * 100, m_strat_exp2['fnr'] * 100]

    plt.bar(x - width / 2, truthful_fnr, width, label='Exp 1: Truthful', color='#16a085', alpha=0.85)
    plt.bar(x + width / 2, manip_fnr, width, label='Exp 2: Manipulated', color='#8e44ad', alpha=0.85)

    plt.ylabel('False Negative Rate (FNR %)', fontsize=12)
    plt.title('Social Implications: Gaming Tax on Qualified Applicants', fontsize=13, fontweight='bold')
    plt.xticks(x, models, fontsize=11)
    plt.ylim([0, max(max(truthful_fnr), max(manip_fnr)) + 12])
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(models)):
        fn1_cnt = m_base_exp1['fn'] if i == 0 else m_strat_exp1['fn']
        fn2_cnt = m_base_exp2['fn'] if i == 0 else m_strat_exp2['fn']
        plt.text(x[i] - width / 2, truthful_fnr[i] + 0.8, f"{truthful_fnr[i]:.1f}%\n({fn1_cnt:.0f})", ha='center', fontsize=9)
        plt.text(x[i] + width / 2, manip_fnr[i] + 0.8, f"{manip_fnr[i]:.1f}%\n({fn2_cnt:.0f})", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gaming_tax_fnr.png"), dpi=200)
    plt.close()

    # 4. Training Loss Convergence Curve
    if len(loss_history) > 0:
        plt.figure(figsize=(7, 4.5))
        plt.plot(loss_history, color='#8e44ad', lw=2.2, label='Worst-Case Strategic Loss')
        plt.xlabel('Subgradient Iteration (t)', fontsize=12)
        plt.ylabel('Regularized Strategic Hinge Risk', fontsize=12)
        plt.title('Strategic Classifier (Algorithm 3) Optimization Convergence', fontsize=13, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=200)
        plt.close()


def run_pipeline(
    csv_path: str,
    reports_dir: str,
    utility_u: float = 0.5,
    n_iter: int = 100,
    learning_rate: float = 0.4,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the complete experimental study and saves all artifacts.
    """
    print("=" * 80)
    print("ONE-SHOT STRATEGIC CLASSIFICATION UNDER UNKNOWN COSTS (ICML 2024)")
    print("Full-Batch Subgradient Method (Algorithm 3) vs Non-Strategic Baseline")
    print("=" * 80)

    # 1. Data Loading & Preprocessing
    print("\n[Step 1] Loading and preprocessing loan approval dataset...")
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_preprocess_loan_data(
        csv_path=csv_path,
        test_size=0.25,
        random_state=random_state
    )
    print(f"  -> Train samples : {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    print(f"  -> Feature count : {X_train.shape[1]}")
    print(f"  -> Class balance : Train Approved (+1): {np.mean(y_train == 1.0)*100:.2f}% | "
          f"Test Approved (+1): {np.mean(y_test == 1.0)*100:.2f}%")

    # 2. Uncertainty Set & Hidden True Cost Matrix
    print("\n[Step 2] Defining Cost Uncertainty Set C and sampling hidden True Cost Sigma*...")
    uncertainty_set = construct_domain_cost_bounds(feature_names)
    Sigma_star, sigma_star = uncertainty_set.sample_hidden_true_cost(random_state=123)

    print("  -> Uncertainty Set C bounds sample:")
    for i in range(min(5, len(feature_names))):
        print(f"     * {feature_names[i]:<28}: sigma in [{uncertainty_set.sigma_min[i]:.2f}, {uncertainty_set.sigma_max[i]:.2f}]")
    print("  -> Hidden True Cost Matrix Sigma* sampled. (Hidden from all training algorithms).")

    # 3. Train Naive Baseline Classifier
    print("\n[Step 3] Training Naive Baseline Classifier (LogisticRegression)...")
    baseline_clf = BaselineClassifier(max_iter=1000, random_state=random_state)
    baseline_clf.fit(X_train, y_train)
    beta_base, b_base = baseline_clf.coef_, baseline_clf.intercept_
    print(f"  -> Baseline fitted. Intercept = {b_base:.4f}, Weight norm ||beta||_2 = {np.linalg.norm(beta_base):.4f}")

    # 4. Train Strategic Classifier (Algorithm 3)
    print("\n[Step 4] Training Minimax Strategic Classifier (Algorithm 3 with Dual Norm Reg)...")
    strategic_clf = StrategicClassifier(
        uncertainty_set=uncertainty_set,
        u_star=utility_u,
        lam=float(np.mean(y_train == 1.0)),
        n_iter=n_iter,
        learning_rate=learning_rate,
        fit_intercept=True,
        verbose=True
    )
    strategic_clf.fit(X_train, y_train)
    beta_strat, b_strat = strategic_clf.coef_, strategic_clf.intercept_
    print(f"  -> Strategic fitted. Intercept = {b_strat:.4f}, Weight norm ||beta||_2 = {np.linalg.norm(beta_strat):.4f}")

    # 5. Experiment 1: Truthful Evaluation
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: NO MISREPORTING (COMPLETELY TRUTHFUL TEST DATA)")
    print("=" * 80)
    m_base_exp1 = calculate_metrics(y_test, baseline_clf.predict(X_test))
    m_strat_exp1 = calculate_metrics(y_test, strategic_clf.predict(X_test))

    print(f"{'Model':<24} | {'Accuracy':<10} | {'FPR':<10} | {'FNR (Gaming Tax)':<18} | {'TPR (Recall)':<12} | {'Precision':<10}")
    print("-" * 96)
    print(f"{'Baseline (Naive)':<24} | {m_base_exp1['accuracy']*100:6.2f}%    | {m_base_exp1['fpr']*100:6.2f}%    | {m_base_exp1['fn']:.0f} ({m_base_exp1['fnr']*100:5.2f}%)       | {m_base_exp1['tpr']*100:6.2f}%       | {m_base_exp1['precision']*100:6.2f}%")
    print(f"{'Strategic (Algorithm 3)':<24} | {m_strat_exp1['accuracy']*100:6.2f}%    | {m_strat_exp1['fpr']*100:6.2f}%    | {m_strat_exp1['fn']:.0f} ({m_strat_exp1['fnr']*100:5.2f}%)       | {m_strat_exp1['tpr']*100:6.2f}%       | {m_strat_exp1['precision']*100:6.2f}%")

    # 6. Experiment 2: Strategic Misreporting
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2: STRATEGIC MISREPORTING (NET UTILITY > 0, Utility u = {utility_u})")
    print("=" * 80)
    simulator = StrategicSimulator(Sigma_star, sigma_star, utility_u=utility_u)

    X_test_manip_base, moved_base, _ = simulator.simulate(X_test, beta_base, b_base)
    X_test_manip_strat, moved_strat, _ = simulator.simulate(X_test, beta_strat, b_strat)

    m_base_exp2 = calculate_metrics(y_test, baseline_clf.predict(X_test_manip_base))
    m_strat_exp2 = calculate_metrics(y_test, strategic_clf.predict(X_test_manip_strat))

    print(f"{'Model':<24} | {'Manipulated':<12} | {'Accuracy':<10} | {'FPR':<10} | {'FNR (Gaming Tax)':<18} | {'TPR':<10} | {'Precision':<10}")
    print("-" * 110)
    print(f"{'Baseline (Naive)':<24} | {np.sum(moved_base):<5} ({np.mean(moved_base)*100:4.1f}%) | {m_base_exp2['accuracy']*100:6.2f}%    | {m_base_exp2['fpr']*100:6.2f}%    | {m_base_exp2['fn']:.0f} ({m_base_exp2['fnr']*100:5.2f}%)       | {m_base_exp2['tpr']*100:6.2f}% | {m_base_exp2['precision']*100:6.2f}%")
    print(f"{'Strategic (Algorithm 3)':<24} | {np.sum(moved_strat):<5} ({np.mean(moved_strat)*100:4.1f}%) | {m_strat_exp2['accuracy']*100:6.2f}%    | {m_strat_exp2['fpr']*100:6.2f}%    | {m_strat_exp2['fn']:.0f} ({m_strat_exp2['fnr']*100:5.2f}%)       | {m_strat_exp2['tpr']*100:6.2f}% | {m_strat_exp2['precision']*100:6.2f}%")

    # 7. Summary & Report Generation
    print("\n" + "=" * 80)
    print("FINAL COMPARATIVE ROBUSTNESS REPORT")
    print("=" * 80)
    base_acc_drop = (m_base_exp1['accuracy'] - m_base_exp2['accuracy']) * 100
    strat_acc_drop = (m_strat_exp1['accuracy'] - m_strat_exp2['accuracy']) * 100
    base_fpr_inc = (m_base_exp2['fpr'] - m_base_exp1['fpr']) * 100
    strat_fpr_inc = (m_strat_exp2['fpr'] - m_strat_exp1['fpr']) * 100

    print(f"Baseline Accuracy Drop : {base_acc_drop:+.2f}%  | Baseline FPR Increase : {base_fpr_inc:+.2f}% (Gamed: {np.sum(moved_base)})")
    print(f"Strategic Accuracy Drop: {strat_acc_drop:+.2f}%  | Strategic FPR Increase: {strat_fpr_inc:+.2f}% (Gamed: {np.sum(moved_strat)})")

    # Save artifacts into reports/
    os.makedirs(reports_dir, exist_ok=True)
    fig_dir = os.path.join(reports_dir, "figures")

    # Generate Markdown Report
    md_report_path = os.path.join(reports_dir, "evaluation_report.md")
    generate_markdown_report(
        m_base_exp1=m_base_exp1,
        m_strat_exp1=m_strat_exp1,
        m_base_exp2=m_base_exp2,
        m_strat_exp2=m_strat_exp2,
        moved_base_count=int(np.sum(moved_base)),
        moved_strat_count=int(np.sum(moved_strat)),
        utility_u=utility_u,
        output_path=md_report_path
    )
    print(f"\n[Artifact] Saved Markdown report to: {md_report_path}")

    # Generate JSON results
    json_path = os.path.join(reports_dir, "experiment_results.json")
    results_dict = {
        "utility_u": utility_u,
        "n_iter": n_iter,
        "learning_rate": learning_rate,
        "experiment_1_truthful": {
            "baseline": m_base_exp1,
            "strategic": m_strat_exp1
        },
        "experiment_2_manipulated": {
            "baseline": m_base_exp2,
            "strategic": m_strat_exp2,
            "moved_baseline": int(np.sum(moved_base)),
            "moved_strategic": int(np.sum(moved_strat))
        },
        "comparisons": {
            "baseline_accuracy_drop": base_acc_drop,
            "strategic_accuracy_drop": strat_acc_drop,
            "baseline_fpr_surge": base_fpr_inc,
            "strategic_fpr_surge": strat_fpr_inc
        }
    }
    save_results_json(results_dict, json_path)
    print(f"[Artifact] Saved JSON metrics to: {json_path}")

    # Generate figures
    plot_evaluation_figures(
        m_base_exp1=m_base_exp1,
        m_strat_exp1=m_strat_exp1,
        m_base_exp2=m_base_exp2,
        m_strat_exp2=m_strat_exp2,
        loss_history=strategic_clf.loss_history_,
        output_dir=fig_dir
    )
    print(f"[Artifact] Saved visualization charts to: {fig_dir}")

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategic Classification Pipeline Runner")
    parser.add_argument("--data", type=str, default="data/loan_data.csv", help="Path to loan approval CSV")
    parser.add_argument("--reports", type=str, default="reports", help="Directory to save report artifacts")
    parser.add_argument("--utility", type=float, default=0.5, help="Loan approval utility payoff u")
    parser.add_argument("--iters", type=int, default=100, help="Subgradient iterations T")
    parser.add_argument("--lr", type=float, default=0.4, help="Learning rate eta")
    args = parser.parse_args()

    # Resolve data path
    if not os.path.exists(args.data):
        alt = os.path.join(parent_dir, "data", "loan_data.csv")
        if os.path.exists(alt):
            args.data = alt

    reports_dir = os.path.join(os.path.abspath(args.reports), "single_run")
    os.makedirs(reports_dir, exist_ok=True)

    run_pipeline(
        csv_path=args.data,
        reports_dir=reports_dir,
        utility_u=args.utility,
        n_iter=args.iters,
        learning_rate=args.lr
    )

