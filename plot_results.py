#!/usr/bin/env python3
"""
Plotting of solver performance for LP/CP-Contest 2023 Problem 1.

Reads results/metrics.csv and produces:

1. Solver runtime vs. board size (saved as results/runtime_vs_size.png).
2. Solver penalty vs. board size (saved as results/penalty_vs_size.png).

Usage:
    python3 plot_results.py
"""

from __future__ import annotations
import os
from typing import NoReturn

import pandas as pd
import matplotlib.pyplot as plt


def plot_runtime(df: pd.DataFrame) -> None:
    """
    Plot solver runtime as a function of board size and save to file.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'solver', 'S', 'time_s'.
    """
    plt.figure()
    for solver in df['solver'].unique():
        sub = df[df['solver'] == solver]
        plt.plot(sub['S'], sub['time_s'], marker='o', label=solver)
    plt.xlabel("Board size S")
    plt.ylabel("Time (s)")
    plt.title("Solver Runtime vs. Board Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/runtime_vs_size.png")
    plt.close()


def plot_penalty(df: pd.DataFrame) -> None:
    """
    Plot solver penalty as a function of board size and save to file.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'solver', 'S', 'penalty'.
    """
    plt.figure()
    for solver in df['solver'].unique():
        sub = df[df['solver'] == solver]
        plt.plot(sub['S'], sub['penalty'], marker='o', label=solver)
    plt.xlabel("Board size S")
    plt.ylabel("Penalty")
    plt.title("Solver Penalty vs. Board Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/penalty_vs_size.png")
    plt.close()


def main() -> NoReturn:
    """
    Load the CSV of results, generate both plots, and exit.
    """
    # ensure results/ directory exists
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv("results/metrics.csv")
    plot_runtime(df)
    plot_penalty(df)

    # Inform the user
    print("Plots saved to results/runtime_vs_size.png and results/penalty_vs_size.png")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
