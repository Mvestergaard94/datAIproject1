#!/usr/bin/env python3
"""
CP-SAT solver stub for LP/CP-Contest Problem 1
Only parses the instance and prints board size for now.
"""
from ortools.sat.python import cp_model
import sys, pathlib, numpy as np

# allow `import common` from the repo root
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import common as C      # <-- no “code.” prefix

def main(inst_path: str, out_path: str):
    S, board = C.read_instance(inst_path)
    print(f"Loaded board of size {S}×{S} with {np.sum(board == 2)} free cells")
    # TODO: add variables, constraints, solve, and write solution
    # For now, just write the input board back out (placeholder)
    C.write_solution(out_path, board.clip(0, 1))  # ensure only 0/1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: solver_cp.py <instance> <output>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

