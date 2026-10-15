"""Compatibility entry point. Implementation: training/ridge.py."""
import importlib
import runpy
import sys

if __name__ == '__main__':
    runpy.run_module('training.ridge', run_name='__main__')
else:
    sys.modules[__name__] = importlib.import_module('training.ridge')
