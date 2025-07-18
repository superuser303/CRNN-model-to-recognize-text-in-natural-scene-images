#!/usr/bin/env python3
import sys
import os

# Add the parent directory to Python path so we can import from 'data'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import auto_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["IIIT5K", "SynthText"])
    args = parser.parse_args()
    
    for name in args.datasets:
        print(f"Preparing dataset: {name}")
        path = auto_data.get_dataset(name)
        print(f"Dataset ready at: {path}")