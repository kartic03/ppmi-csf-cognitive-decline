"""
Phase 2 package: leak-safe nested-CV harness for PD cognitive-decline prediction.

Downstream tasks (B3, B4) can import the harness entry points without
replicating the importlib dance for digit-prefixed modules:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cv import nested_cv, fit_eb_params, eb_slopes, load_moca

Or, if the parent of src/phase2 is on sys.path:

    from phase2.cv import nested_cv, fit_eb_params, eb_slopes, load_moca

cv.py handles loading 01_outcome.py (digit-prefixed) internally.
"""
