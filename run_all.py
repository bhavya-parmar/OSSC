"""
Master CLI Entrypoint for One-Shot Strategic Classification.

Usage:
    python run_all.py
    python run_all.py --utility 0.5 --iters 100 --seeds 42 101 2024 7 99
"""

import os
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_ossc_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.run_master_pipeline import main

if __name__ == "__main__":
    main()
