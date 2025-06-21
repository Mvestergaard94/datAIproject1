#!/usr/bin/env python3
"""
CP-SAT solver for LP/CP-Contest 2023 – Problem 1 (“Fair exercise allocation”).

Features
--------
* Row / column balance (exactly ⌊S / 2⌋ ones).
* Row / column **uniqueness** – no two identical rows or columns.
* Sliding 3-window **mix** – no 000 or 111 in any row/col window of length 3.
* All code is type-annotated and documented.
* Optional **verbose** “spy” mode prints model size & solve time.

Usage
-----
    python solver_cp.py <instance> <output> [--verbose]

Example
-------
    python solver_cp.py data/p1/instance.1.in results/out.txt --verbose
"""

from __future__ import annotations

import sys
import time
import pathlib
from typing import Dict, Tuple

import numpy as np
from ortools.sat.python import cp_model

# --------------------------------------------------------------------------- #
#  Project-local import: common.py contains the parser / writer utilities.    #
# --------------------------------------------------------------------------- #
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402  (import after path tweak)


def build_and_solve(
    S: int,
    board: np.ndarray,
    time_limit: int = 30,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build the OR-Tools CP-SAT model and return a feasible 0/1 NumPy board.

    Parameters
    ----------
    S : int
        Board size (rows = cols = S).
    board : np.ndarray
        S × S array with values {0, 1, 2}.  2 means “free / decision var”.
    time_limit : int, default 30
        Solver wall-clock time limit in seconds.
    verbose : bool, default False
        If True, print model statistics and solution info.

    Returns
    -------
    np.ndarray
        An S × S array of 0/1 values that satisfies all constraints.

    Raises
    ------
    RuntimeError
        If no feasible solution is found within *time_limit*.
    """
    m: cp_model.CpModel = cp_model.CpModel()

    # ---------------- decision variables ---------------------------------- #
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for i in range(S):
        for j in range(S):
            if board[i, j] == 2:
                x[i, j] = m.NewBoolVar(f"x_{i}_{j}")
            else:
                # Fixed cells become constant literals in the model
                x[i, j] = m.NewConstant(int(board[i, j]))

    # ---------------- row / column balance -------------------------------- #
    half: int = S // 2
    for i in range(S):
        m.Add(sum(x[i, j] for j in range(S)) == half)
    for j in range(S):
        m.Add(sum(x[i, j] for i in range(S)) == half)

    # ---------------- uniqueness (AllDifferent on row/col codes) ---------- #
    row_codes = []
    for i in range(S):
        code = m.NewIntVar(0, 2**S - 1, f"rowcode_{i}")
        m.Add(code == sum(x[i, j] * (1 << j) for j in range(S)))
        row_codes.append(code)
    m.AddAllDifferent(row_codes)

    col_codes = []
    for j in range(S):
        code = m.NewIntVar(0, 2**S - 1, f"colcode_{j}")
        m.Add(code == sum(x[i, j] * (1 << i) for i in range(S)))
        col_codes.append(code)
    m.AddAllDifferent(col_codes)

    # ---------------- 3-window mix constraint ----------------------------- #
    # No window of length 3 is allowed to sum to 0 or 3.
    for i in range(S):
        for j in range(S - 2):
            win = [x[i, k] for k in range(j, j + 3)]
            s = m.NewIntVar(1, 2, f"r{i}_{j}_mix")  # allowed sums: 1 or 2
            m.Add(s == sum(win))
    for j in range(S):
        for i in range(S - 2):
            win = [x[k, j] for k in range(i, i + 3)]
            s = m.NewIntVar(1, 2, f"c{j}_{i}_mix")
            m.Add(s == sum(win))

    # ---------------- solve ------------------------------------------------ #
    m.Maximize(0)  # feasibility only
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit

    if verbose:
        print(f"[CP-SAT] Variables: {m.NumVariables()}  Constraints: {m.NumConstraints()}")
        t0 = time.time()

    status = solver.Solve(m)

    if verbose:
        print(f"[CP-SAT] Status: {solver.StatusName(status)}  Time: {solver.WallTime():.3f}s\n")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT: no feasible solution in time limit")

    sol = np.zeros_like(board)
    for i in range(S):
        for j in range(S):
            sol[i, j] = solver.Value(x[i, j])

    return sol


# --------------------------------------------------------------------------- #
#  CLI entry-point                                                            #
# --------------------------------------------------------------------------- #
def main() -> None:
    """Parse CLI args, solve the instance, and write the solution file."""
    if not (3 <= len(sys.argv) <= 4):
        print("Usage: solver_cp.py <instance> <output> [--verbose]")
        sys.exit(1)

    inst_path: str = sys.argv[1]
    out_path: str = sys.argv[2]
    verbose: bool = len(sys.argv) == 4 and sys.argv[3] == "--verbose"

    S, board = C.read_instance(inst_path)
    print(f"Loaded board {S}×{S}  free cells={np.sum(board == 2)}")

    sol = build_and_solve(S, board, verbose=verbose)
    C.write_solution(out_path, sol)


if __name__ == "__main__":  # pragma: no cover
    main()
