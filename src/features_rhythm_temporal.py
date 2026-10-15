"""Compatibility entry point. Implementation: features/rhythm_windows.py."""
import importlib
import runpy
import sys

if __name__ == '__main__':
    runpy.run_module('features.rhythm_windows', run_name='__main__')
else:
    sys.modules[__name__] = importlib.import_module('features.rhythm_windows')
