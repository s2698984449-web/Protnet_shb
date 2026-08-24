"""Wrapper: invoke eval.py --seeds (multi-seed final eval) from run_experiment.py.

This module exists solely to satisfy YAML references like:
    script: final_eval.py
It simply delegates to eval.py:main().
"""
import sys
import os

# Ensure project root is on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if __name__ == '__main__':
    from scripts.eval import main as _eval_main
    sys.exit(_eval_main())
