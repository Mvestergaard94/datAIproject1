#!/usr/bin/env python3
"""
Simulated‐Annealing solver for LP/CP‐Contest 2023 – Problem 1 (“Fair exercise allocation”).

Keeps all constraints as penalties; energy == 0 ⇒ feasible.

Example
-------
>>> # generate a tiny 4×4 instance with half fixed and half free:
>>> import numpy as np
>>> from code.sa.solver_sa import anneal, energy
>>> board = 2 * np.ones((4, 4), dtype=int)
>>> # force a single 1 in each row to make it solvable:
>>> board[0, 0] = 1
>>> board[1, 1] = 1
>>> board[2, 2] = 1
>>> board[3, 3] = 1
>>> sol = anneal(board, seconds=0.1)
>>> assert energy(sol) == 0
"""

from __future__ import annotations

import sys
import time
import pathlib
import random
from typing import Tuple

import numpy as np

# make common.py importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

import logging

# ensure log directory exists
pathlib.Path("logs/sa").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename="logs/sa/instance.log",
    format="%(message)s",
    level=logging.INFO
)


def random_fill(board: np.ndarray) -> np.ndarray:
    """
    Fill every '2' cell so each row/column has exactly S//2 ones.

    Parameters
    ----------
    board : np.ndarray
        S×S array with entries in {0,1,2} (2 = “free”).

    Returns
    -------
    np.ndarray
        S×S array with 0/1 in all positions.

    >>> import numpy as np
    >>> b = np.array([[2, 1], [0, 2]])
    >>> # S=2, half=1: first row needs 0 more 1s, second row needs 1 one:
    >>> out = random_fill(b)
    >>> out.shape == (2, 2)
    True
    >>> set(out[:,0]).issubset({0,1})
    True
    """
    S = board.shape[0]
    half = S // 2
    out = board.copy()

    # fill rows
    for i in range(S):
        free = np.where(out[i] == 2)[0]
        need = half - int((out[i] == 1).sum())
        choice = np.random.choice(free, need, replace=False)
        out[i, choice] = 1
        other = [j for j in free if j not in set(choice)]
        out[i, other] = 0

    # fix columns if needed
    for j in range(S):
        diff = int(out[:, j].sum()) - half
        if diff != 0:
            free = np.where(board[:, j] == 2)[0]
            flip = np.random.choice(free, abs(diff), replace=False)
            for i in flip:
                out[i, j] = 1 - out[i, j]

    return out


def energy(M: np.ndarray) -> int:
    """
    Compute the penalty of board M.

    Zero means feasible. Higher = more constraint‐violations.

    >>> import numpy as np
    >>> M = np.array([[1,1,0,0],
    ...               [0,0,1,1],
    ...               [1,0,1,0],
    ...               [0,1,0,1]])
    >>> energy(M)
    0
    """
    S = M.shape[0]
    half = S // 2
    e = 0

    # balance rows & cols
    e += abs((M.sum(axis=1) - half)).sum()
    e += abs((M.sum(axis=0) - half)).sum()

    # uniqueness
    e += (S - len({tuple(r) for r in M})) * S
    e += (S - len({tuple(c) for c in M.T})) * S

    # sliding-3 windows
    for i in range(S):
        for j in range(S - 2):
            s = int(M[i, j : j + 3].sum())
            if s in (0, 3):
                e += 1
    for j in range(S):
        for i in range(S - 2):
            s = int(M[i : i + 3, j].sum())
            if s in (0, 3):
                e += 1

    return int(e)


def neighbour(M: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """
    Return a new board by swapping two mutable cells in one random row.

    Parameters
    ----------
    M : np.ndarray
        Current board (0/1).
    fixed : np.ndarray
        Boolean mask of same shape: True if cell is fixed.

    Returns
    -------
    np.ndarray
        New board with one swap applied.
    """
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


def anneal(board: np.ndarray, seconds: float = 20.0, verbose: bool = False) -> np.ndarray:
    """
    Run Simulated Annealing for up to `seconds`, return best found.

    Parameters
    ----------
    board : np.ndarray
        S×S array with {0,1,2}, where 2 means decision variable.
    seconds : float
        Wall‐clock time budget.
    verbose : bool
        If True, logs temperature & best energy.

    Returns
    -------
    np.ndarray
        S×S array of 0/1 satisfying (hopefully) energy == 0.

    Example
    -------
    >>> import numpy as np
    >>> b = 2 * np.ones((4,4), dtype=int)
    >>> out = anneal(b, seconds=0.01)
    >>> isinstance(out, np.ndarray)
    True
    """
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

    while time.time() < t_end and bestE > 0:
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


def main() -> None:
    """
    CLI entry point for solver_sa.py.

    Usage:
      python solver_sa.py <instance> <output> [--verbose]

    Reads an instance file, runs SA for 40s, writes solution, prints stats, and logs penalty+time.
    """
    if not (3 <= len(sys.argv) <= 4):
        print("Usage: solver_sa.py <instance> <output> [--verbose]")
        sys.exit(1)

    inst_path, out_path = sys.argv[1], sys.argv[2]
    verbose = (len(sys.argv) == 4 and sys.argv[3] == "--verbose")

    # read and report
    S, board = C.read_instance(inst_path)
    print(f"Loaded board {S}×{S}  free cells={(board == 2).sum()}")

    # run SA
    start = time.time()
    sol = anneal(board, seconds=40.0, verbose=verbose)
    elapsed = time.time() - start

    # write solution
    C.write_solution(out_path, sol)

    # compute and log final penalty
    pen = energy(sol)
    logging.info(f"PENALTY: {pen}")
    logging.info(f"TIME:    {elapsed:.3f}s")

    # echo back to user
    print(f"TIME: {elapsed:.3f}s   PENALTY: {pen}")


if __name__ == "__main__":
    main()
