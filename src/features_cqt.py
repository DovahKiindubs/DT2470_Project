"""Compatibility entry point. Implementation: features/cqt.py."""
import importlib
import runpy
import sys

if __name__ == '__main__':
    runpy.run_module('features.cqt', run_name='__main__')
else:
    sys.modules[__name__] = importlib.import_module('features.cqt')
