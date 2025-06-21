# datAIproject1

# LP/CP Contest 2023 – Problem 1

Two solvers are provided:

| Folder | File | Method | Notes |
|--------|------|--------|-------|
| `code/cp_sat` | `solver_cp.py` | OR-Tools CP-SAT | deterministic, <1 s |
| `code/sa`     | `solver_sa.py` | Simulated Annealing | 40 s wall-time |

## Reproduce results

```bash
# 1 — create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2 — install exact dependencies
pip install -r requirements.txt
