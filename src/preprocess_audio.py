"""Compatibility entry point. Implementation: data/preprocess_audio.py."""
import importlib
import runpy
import sys

if __name__ == '__main__':
    runpy.run_module('data.preprocess_audio', run_name='__main__')
else:
    sys.modules[__name__] = importlib.import_module('data.preprocess_audio')
