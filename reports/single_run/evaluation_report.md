# Experimental Robustness Report: Strategic Classification Under Unknown Costs

**Paper Reference**: *One-Shot Strategic Classification Under Unknown Costs (ICML 2024)*  
**Dataset**: Real-World Loan Approval Dataset  
**Utility Setting**: u = 0.5

---

## 1. Experiment Overview

- **Experiment 1 (Truthful Baseline)**: Classifiers evaluated on pristine, unmanipulated test data.
- **Experiment 2 (Strategic Gaming)**: Test applicants dynamically respond to the deployed classifier based on rational utility maximization under the hidden True Cost Matrix Sigma*:
  Delta*(x) = argmax_{x'} [ 1{beta^T x' + b >= 0} * u - c(x, x') ]

---

## 2. Quantitative Performance Summary

| Metric | Naive Baseline (Logistic Regression) | Strategic Classifier (Algorithm 3) | Performance / Robustness Impact |
| :--- | :---: | :---: | :--- |
| **Truthful Accuracy (Exp 1)** | 89.97% | 88.61% | Baseline +1.36% on clean data |
| **Manipulated Accuracy (Exp 2)** | 87.61% | **89.25%** | **Strategic outperforms Baseline by +1.64%** |
| **Accuracy Degradation** | **+2.36%** | **-0.64%** | **Strategic suffers minimal degradation** |
| **Truthful FPR (Exp 1)** | 5.62% | 2.77% | Strategic initiates with lower FPR |
| **Manipulated FPR (Exp 2)** | **12.80%** | **4.25%** | **Baseline FPR surges dramatically!** |
| **FPR Surge (Delta FPR)** | **+7.18%** | **+1.49%** | **Strategic is significantly more resistant** |
| **Truthful False Negatives (FN count)** | 636 (25.44% FNR) | 1039 (41.56% FNR) | Conservative boundary increases initial FN |
| **Manipulated False Negatives (FN count)** | 274 (10.96% FNR) | 837 (33.48% FNR) | Gaming Tax: qualified applicants denied |
| **Gaming Applicants** | **990 applicants** | **332 applicants** | **66.5% reduction in gaming** |

---

## 3. Detailed Takeaways & Social Implications (Gaming Tax)

1. **Institutional Robustness (Vulnerability of Standard Learning)**:
   The standard Logistic Regression model exhibits catastrophic failure under strategic behavior. Its False Positive Rate increases from 5.62% to 12.80% (+7.18% surge), because it relies heavily on easy-to-manipulate features.

2. **Minimax Defense via Algorithm 3**:
   The strategic classifier de-sensitizes low-cost dimensions by anticipating the worst-case cost matrix in C. Only 332 applicants could afford to game it (compared to 990 for the baseline), keeping FPR at 4.25% while preserving 89.25% accuracy.

3. **The 'Gaming Tax' on Qualified Borrowers (Social Cost)**:
   To protect the lender against fraud, the strategic model adopts a more conservative boundary, resulting in a higher False Negative count (1039 vs 636 in Exp 1).
   Qualified borrowers who look borderline are denied unless they exert costly manipulation effort. For disadvantaged applicants with high personal modification costs, this creates an unfair barrier ('Gaming Tax', Milli et al., 2019).
