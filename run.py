"""
Entry point CLI for running the Strategic Classification pipeline from the OSSC repository root.

Usage:
    python run.py
    python run.py --utility 0.5 --iters 100 --lr 0.4
"""

import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_experiments import run_pipeline

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "loan_data.csv")
    reports_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

    run_pipeline(
        csv_path=csv_file,
        reports_dir=reports_folder,
        utility_u=0.5,
        n_iter=100,
        learning_rate=0.4
    )

