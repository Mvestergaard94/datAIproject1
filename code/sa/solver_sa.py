#!/usr/bin/env python3
"""
Simulated-Annealing solver for LP/CP-Contest 2023 – Problem 1.

* Keeps all constraints as penalties; score 0 ⇢ feasible.
* “Spy” mode (--verbose) prints energy trajectory & best score.
"""

from __future__ import annotations
import time                      # ← add this RIGHT AFTER the other imports
import random
import numpy as np
import sys
import pathlib
from typing import Tuple

# --------------------------------------------------------------------------- #
#  Local parser / writer                                                      #
# --------------------------------------------------------------------------- #
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402  (import after path tweak)


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def random_fill(board: np.ndarray) -> np.ndarray:
    """Fill every '2' cell so each row and column has S/2 ones."""
    S = board.shape[0]
    half = S // 2
    out = board.copy()

    # --- rows ---
    for i in range(S):
        free = np.where(out[i] == 2)[0]
        need = half - np.sum(out[i] == 1)
        choice = np.random.choice(free, need, replace=False)
        out[i, choice] = 1
        out[i, free[~np.isin(free, choice)]] = 0

    # --- columns (simple fix-up if imbalance happened) ---
    for j in range(S):
        col = out[:, j]
        diff = np.sum(col) - half
        if diff:  # positive ⇒ too many 1s
            free = np.where(board[:, j] == 2)[0]  # only mutable cells
            flip = np.random.choice(free, abs(diff), replace=False)
            out[flip, j] = 1 - out[flip, j]
    return out


def energy(M: np.ndarray) -> int:
    """Penalty; 0 means all constraints satisfied."""
    S = M.shape[0]
    half = S // 2
    e = 0

    # balance
    e += np.abs(np.sum(M, axis=1) - half).sum()
    e += np.abs(np.sum(M, axis=0) - half).sum()

    # uniqueness
    e += (S - len({tuple(r) for r in M})) * S
    e += (S - len({tuple(c) for c in M.T})) * S

    # 3-window
    for i in range(S):
        for j in range(S - 2):
            s = M[i, j : j + 3].sum()
            if s in (0, 3):
                e += 1
    for j in range(S):
        for i in range(S - 2):
            s = M[i : i + 3, j].sum()
            if s in (0, 3):
                e += 1
    return int(e)


def neighbour(M: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Swap two differing mutable cells in the same row."""
    S = M.shape[0]
    while True:
        i = random.randrange(S)
        free = np.where(~fixed[i])[0]
        if len(free) < 2:
            continue
        j1, j2 = np.random.choice(free, 2, replace=False)
        if M[i, j1] != M[i, j2]:
            N = M.copy()
            N[i, j1], N[i, j2] = N[i, j2], N[i, j1]
            return N


def anneal(board: np.ndarray, seconds: int = 20, verbose: bool = False) -> np.ndarray:
    """Run SA for *seconds* wall-time and return the best board found."""
    S = board.shape[0]
    fixed = board != 2
    cur = random_fill(board)
    best = cur.copy()
    E = energy(cur)
    bestE = E

    T = 5.0
    alpha = 0.95
    t_end = time.time() + seconds

    if verbose:
        print(f"[SA] start energy = {E}")

    while time.time() < t_end and bestE:
        for _ in range(300):
            nxt = neighbour(cur, fixed)
            dE = energy(nxt) - E
            if dE < 0 or random.random() < np.exp(-dE / T):
                cur, E = nxt, E + dE
                if E < bestE:
                    best, bestE = cur.copy(), E
        T *= alpha
        if verbose:
            print(f"[SA] T={T:.3f}  bestE={bestE}", flush=True)

    return best


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    if not (3 <= len(sys.argv) <= 4):
        print("Usage: solver_sa.py <instance> <output> [--verbose]")
        sys.exit(1)

    inst, out = sys.argv[1:3]
    verbose = len(sys.argv) == 4 and sys.argv[3] == "--verbose"

    S, board = C.read_instance(inst)
    print(f"Loaded board {S}×{S}  free cells={np.sum(board == 2)}")

    sol = anneal(board, seconds=40, verbose=verbose)
    C.write_solution(out, sol)


if __name__ == "__main__":  # pragma: no cover
    main()
