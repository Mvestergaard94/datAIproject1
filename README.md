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

# 3 — run SA solver on instance 1 (40 s timeout)
python code/sa/solver_sa.py data/p1/instance.1.in results/sa_1.out

# 4 — validate with contest checker
python AIproject/lpcp-contest-2023/checker.py 1 1 < results/sa_1.out

# 5 — run CP-SAT solver on instance 1 (≈1 s)
python code/cp_sat/solver_cp.py data/p1/instance.1.in results/cp_1.out

# 6 — validate CP-SAT solution
python AIproject/lpcp-contest-2023/checker.py 1 1 < results/cp_1.out

# 7 — batch-run & package all 1–5 SA outputs
for i in {1..5}; do
  python code/sa/solver_sa.py data/p1/instance.$i.in results/sa_$i.out
done
mkdir -p artifacts
zip -j artifacts/solutions.zip results/sa_*.out

# 8 — inspect ZIP
unzip -l artifacts/solutions.zip
