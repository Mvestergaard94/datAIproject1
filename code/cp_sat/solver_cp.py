#!/usr/bin/env python3
"""
CP-SAT solver for LP/CP-Contest 2023 – Problem 1 (“Fair exercise allocation”)

Builds the full model: row/col balance, uniqueness, 3-cell window mix.
"""

from ortools.sat.python import cp_model
import sys, pathlib, numpy as np

# allow "import common" (parser/writer)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import common as C        # noqa


def build_and_solve(S: int, board: np.ndarray, time_limit=30):
    m = cp_model.CpModel()

    # --------------- decision variables ----------------
    x = {}
    for i in range(S):
        for j in range(S):
            if board[i, j] == 2:
                x[i, j] = m.NewBoolVar(f"x_{i}_{j}")
            else:
                x[i, j] = m.NewConstant(int(board[i, j]))

    # --------------- row / column balance --------------
    half = S // 2
    for i in range(S):
        m.Add(sum(x[i, j] for j in range(S)) == half)
    for j in range(S):
        m.Add(sum(x[i, j] for i in range(S)) == half)

    # --------------- uniqueness constraints ------------
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

    # --------------- 3-cell sliding-window mix ---------
    # no window of length 3 may sum to 0 or 3 (all-equal)
    for i in range(S):
        for j in range(S - 2):
            win = [x[i, k] for k in range(j, j + 3)]
            s = m.NewIntVar(1, 2, f"r{i}_{j}_mix")   # allowed sums 1 or 2
            m.Add(s == sum(win))
    for j in range(S):
        for i in range(S - 2):
            win = [x[k, j] for k in range(i, i + 3)]
            s = m.NewIntVar(1, 2, f"c{j}_{i}_mix")
            m.Add(s == sum(win))

    # --------------- solve -----------------------------
    m.Maximize(0)                         # feasibility only
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(m)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT: no feasible solution in time limit")

    sol = np.zeros_like(board)
    for i in range(S):
        for j in range(S):
            sol[i, j] = solver.Value(x[i, j])
    return sol


def main(inst_path: str, out_path: str):
    S, board = C.read_instance(inst_path)
    print(f"Loaded board {S}×{S}  free cells={np.sum(board == 2)}")
    sol = build_and_solve(S, board)
    C.write_solution(out_path, sol)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: solver_cp.py <instance> <output>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
