"""
Tool for visualizing the results of multiple Authorship Verification
evaluations.
Input:
JSON-file
{
    results: {
        measure1: number,
        measure2: number,
        ...
    }
    folds: int (optional)
}
"""
import argparse
import time
import os

import pandas as pd
import matplotlib as np
import numpy as plt


def now(): return time.strftime("%Y-%m-%d_%H-%M-%S")


def main():
    parser = argparse.ArgumentParser(
        prog="plot",
        description="Plot Authorship Verification evaluation results",
        add_help=True)
    parser.add_argument('--input',
                        '-i',
                        required=True,
                        help='Path to directory of evaluation results')
    args = parser.parse_args()

    out_path = os.path.join('data', f'plot_{now()}')
    os.makedirs(out_path, exist_ok=True)


if __name__ == "__main__":
    main()
