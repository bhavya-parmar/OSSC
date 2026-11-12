"""
Legacy test runner redirecting to the modular test suite in tests/.
"""

import os
import sys
import pytest

if __name__ == "__main__":
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    print(f"Running test suite in {test_dir} ...")
    exit_code = pytest.main([test_dir, "-v"])
    sys.exit(exit_code)
