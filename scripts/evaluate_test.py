"""Evaluate all nine frozen rerun_v1 models on test without training."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiments.evaluate_test import main

if __name__=='__main__':main()
