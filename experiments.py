#!/usr/bin/env python3
"""
Run SA and CP‐SAT on all Problem 1 instances and record time, penalty, and model size.

Writes out results/metrics.csv with columns:
  solver, instance, S, free_cells, time_s, penalty, vars, cons

Usage:
    python3 experiments.py

After it finishes (≈40 s×5 for SA), you can peek with:
    head -n5 results/metrics.csv
"""

import csv
import subprocess
import re
import sys
from typing import List, Tuple, Optional
from code.common import read_instance  # noqa: E402

def run_experiments() -> None:
    """
    Executes each solver on each instance and writes out metrics/metrics.csv.

    For each run extracts:
      - time_s   from “TIME: …s”
      - penalty  from “PENALTY: …”
      - vars, cons (only for CP-SAT) from “[CP-SAT] Vars: …  Cs: …”
    """
    INSTANCES: List[str] = [f"data/p1/instance.{i}.in" for i in range(1, 6)]
    SOLVERS: List[Tuple[str, List[str]]] = [
        ("SA", [sys.executable, "code/sa/solver_sa.py"]),
        ("CP", [sys.executable, "code/cp_sat/solver_cp.py"]),
    ]

    with open("results/metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver", "instance", "S", "free_cells", "time_s", "penalty", "vars", "cons"])
        for inst in INSTANCES:
            S, board = read_instance(inst)
            free_cells: int = int((board == 2).sum())
            for name, cmd_base in SOLVERS:
                cmd = cmd_base + [inst, "/dev/null", "--verbose"]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                out = proc.stdout + proc.stderr

                # extract vars & constraints (CP only)
                m_vars = re.search(r"Vars:\s*(\d+)", out)
                m_cons = re.search(r"Cs:\s*(\d+)", out)
                vars_count: Optional[int] = int(m_vars.group(1)) if m_vars else None
                cons_count: Optional[int] = int(m_cons.group(1)) if m_cons else None

                # extract time & penalty
                m_time = re.search(r"TIME: ([0-9.]+)s", out)
                m_pen  = re.search(r"PENALTY: (\d+)", out)
                time_s: Optional[float] = float(m_time.group(1)) if m_time else None
                penalty: Optional[int]  = int(m_pen.group(1)) if m_pen else None

                writer.writerow([
                    name,
                    inst,
                    S,
                    free_cells,
                    time_s,
                    penalty,
                    vars_count,
                    cons_count,
                ])

if __name__ == "__main__":
    run_experiments()
