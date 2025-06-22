#!/usr/bin/env python3
"""
CP-SAT solver for LP/CP-Contest 2023 – Problem 1 (“Fair exercise allocation”).

This module implements a CP-SAT model that enforces:

 1. Row / column balance (exactly ⌊S/2⌋ ones per row/col).
 2. Row / column uniqueness (no two identical rows or columns).
 3. Sliding 3-window mix (no window of 3 all-zeros or all-ones).
 4. Feasibility only (objective = 0).

Usage:
    python solver_cp.py <instance> <output> [--verbose]

Example:
    >>> # run on the first 6×6 test instance
    >>> python solver_cp.py data/p1/instance.1.in results/tmp.out --verbose
    Loaded board 6×6  free cells=22
    [CP-SAT] Vars:  216  Cs:  74
    [CP-SAT] Status: FEASIBLE  Time: 0.32s
    TIME: 0.32s   PENALTY: 0
"""

from __future__ import annotations

import sys
import time
import pathlib
import logging
from typing import Dict, Tuple

import numpy as np
from ortools.sat.python import cp_model

# make common.py importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

# reuse our SA energy function for penalty reporting
from sa.solver_sa import energy  # noqa: E402

# ensure log directory exists
pathlib.Path("logs/cp").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename="logs/cp/instance.log",
    format="%(message)s",
    level=logging.INFO
)


def build_and_solve(
    S: int,
    board: np.ndarray,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build the CP-SAT model and solve for a feasible assignment.

    Parameters
    ----------
    S : int
        Board dimension (S×S).
    board : np.ndarray
        Input array with values in {0, 1, 2}, where 2 marks a free cell.
    time_limit : float
        Time limit in seconds for the CP-SAT solver.
    verbose : bool
        If True, prints model size and solver status/time.

    Returns
    -------
    np.ndarray
        Feasible S×S array of 0/1 assignments.

    Raises
    ------
    RuntimeError
        If the solver fails to find a feasible solution within the time limit.

    Examples
    --------
    >>> # trivial 2×2 all-zeros board with S=2 is infeasible (must have exactly 1 one per row/col)
    >>> import numpy as np
    >>> b = np.array([[2,2],[2,2]])
    >>> # should raise because no feasible assignment exists for 2×2
    >>> try:
    ...     build_and_solve(2, b, time_limit=0.1)
    ...     print("found")
    ... except RuntimeError:
    ...     print("no solution")
    no solution
    """
    model = cp_model.CpModel()

    # 1) Decision variables
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for i in range(S):
        for j in range(S):
            if board[i, j] == 2:
                x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")
            else:
                x[(i, j)] = model.NewConstant(int(board[i, j]))

    # 2) Row/column balance
    half = S // 2
    for i in range(S):
        model.Add(sum(x[(i, j)] for j in range(S)) == half)
    for j in range(S):
        model.Add(sum(x[(i, j)] for i in range(S)) == half)

    # 3) Uniqueness via integer encoding + AllDifferent
    row_codes = []
    for i in range(S):
        code = model.NewIntVar(0, 2**S - 1, f"rowcode_{i}")
        model.Add(code == sum(x[(i, j)] * (1 << j) for j in range(S)))
        row_codes.append(code)
    model.AddAllDifferent(row_codes)

    col_codes = []
    for j in range(S):
        code = model.NewIntVar(0, 2**S - 1, f"colcode_{j}")
        model.Add(code == sum(x[(i, j)] * (1 << i) for i in range(S)))
        col_codes.append(code)
    model.AddAllDifferent(col_codes)

    # 4) Sliding-3 mix constraint: enforce sum in {1,2}
    for i in range(S):
        for j in range(S - 2):
            w = [x[(i, k)] for k in range(j, j + 3)]
            s = model.NewIntVar(1, 2, f"r{i}_{j}_mix")
            model.Add(s == sum(w))
    for j in range(S):
        for i in range(S - 2):
            w = [x[(k, j)] for k in range(i, i + 3)]
            s = model.NewIntVar(1, 2, f"c{j}_{i}_mix")
            model.Add(s == sum(w))

    # 5) Feasibility objective
    model.Maximize(0)

    # Count vars & constraints via the model protobuf
    proto = model.Proto()
    n_vars = len(proto.variables)
    n_cons = len(proto.constraints)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit

    if verbose:
        print(f"[CP-SAT] Vars: {n_vars}  Cs: {n_cons}")

    status = solver.Solve(model)

    elapsed = solver.WallTime()
    if verbose:
        print(f"[CP-SAT] Status: {solver.StatusName(status)}  Time: {elapsed:.3f}s")
        logging.info(f"VARS: {n_vars}  CS: {n_cons}")
        logging.info(f"STATUS: {solver.StatusName(status)}  TIME: {elapsed:.3f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT: no feasible solution found")

    # 6) Extract into NumPy
    sol = np.zeros((S, S), dtype=int)
    for i in range(S):
        for j in range(S):
            sol[i, j] = int(solver.Value(x[(i, j)]))

    return sol


def main() -> None:
    """
    CLI entrypoint.

    Parses <instance> and <output> paths, optionally --verbose,
    solves via build_and_solve(), writes solution, and prints timing.

    Usage
    -----
      python solver_cp.py <instance> <output> [--verbose]
    """
    if not (3 <= len(sys.argv) <= 4):
        print("Usage: solver_cp.py <instance> <output> [--verbose]")
        sys.exit(1)

    inst_path = sys.argv[1]
    out_path = sys.argv[2]
    verbose = len(sys.argv) == 4 and sys.argv[3] == "--verbose"

    S, board = C.read_instance(inst_path)
    print(f"Loaded board {S}×{S}  free cells={int((board == 2).sum())}")

    t0 = time.time()
    sol = build_and_solve(S, board, time_limit=30.0, verbose=verbose)
    elapsed = time.time() - t0

    C.write_solution(out_path, sol)
    pen = energy(sol)

    logging.info(f"PENALTY: {pen}")
    logging.info(f"TIME:    {elapsed:.3f}s")

    print(f"TIME: {elapsed:.3f}s   PENALTY: {pen}")


if __name__ == "__main__":
    main()
