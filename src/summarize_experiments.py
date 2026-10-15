"""Compatibility entry point. Implementation: reporting/legacy_registry.py."""
import importlib
import runpy
import sys

if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise SystemExit('Use python scripts/run_experiments.py --summarize for the current suite. '
                         'The legacy build API remains available for explicit historical work.')
    from experiments.suite import read_json, summarize, CATALOGUE
    summarize(read_json(CATALOGUE))
else:
    sys.modules[__name__] = importlib.import_module('reporting.legacy_registry')
