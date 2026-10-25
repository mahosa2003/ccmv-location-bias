# Code and results for "Location bias and encoding geometry in metaheuristic benchmarks for cardinality-constrained portfolio optimization"

M. Abdelsalam, H. Diab (AASTMT). Python code, raw results and figures of the paper.

## Contents

| Path | What it is |
|---|---|
| `esdo/core.py` | ESDO (port of the 2023 MATLAB code with two coding slips corrected) and the revised ESDO-R; `Counted`, which enforces exact evaluation budgets |
| `esdo/algos.py` | One interface to the eleven optimizers (GWO, HHO, WOA, SCA, PSO, DE, GA and L-SHADE from MEALPY 3.0.2; BIPOP-CMA-ES from pycma; ESDO and ESDO-R) |
| `esdo/portfolio.py` | CCMV model (Chang et al., 2000) with the top-K encoding, OR-Library loader, MPE metric, exact model for SCIP |
| `esdo/cec17.py` | Not used in this paper; imported by `analyze.py` for function names only |
| `bench_center.py` | Centre test: optimum at the centre of [-100, 100]^30 or shifted |
| `bench_origin.py` | Five-location test in [0, 200]^30: origin, opposite corner, off-diagonal corner, centre, random point |
| `bench_port.py` | CCMV with the direct encoding on OR-Library port1-port5 (K = 10, 20 values of lambda); `--budget m` gives m*n evaluations |
| `bench_mirror.py` | The same problems with the mirrored (`--mode mirror`) or randomly reflected (`--mode corner`) encoding, same seeds |
| `bench_mechanism.py` | Mean coordinate of the evaluated points over a run (where each optimizer searches) |
| `exact_port.py` | SCIP solutions of the 100 frontier points (60 s each) |
| `exact_strong.py` | Strengthened model (diagonal perspective reformulation) for the points left open |
| `analyze.py`, `analyze_p1.py` | Statistics and figures; `analyze_p1.py` writes `results_p1/summary_p1.json` and `figures_p1/` |
| `run_queue.sh`, `queue_*.txt` | Time-boxed runner used for the long jobs (each call stops starting new commands after a set number of minutes) |
| `data/orlib/` | OR-Library port1-5 and portef1-5 (Beasley, 1990) |
| `results_p1/` | Raw results, one JSON line per run, and `summary_p1.json` |
| `figures_p1/` | Figures of the paper (PNG and vector PDF) |
| `requirements.txt`, `LICENSE`, `CITATION.cff` | Python requirements, licence and citation metadata (read by GitHub and Zenodo) |

## Requirements

Python 3.10 or later with numpy, scipy, pandas, matplotlib, mealpy 3.0.2, cma 4.5.0, pyscipopt (SCIP 10) and, for
`exact_strong.py`, cvxpy with the Clarabel solver (`pip install -r requirements.txt`). The results were produced with
Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, MEALPY 3.0.2, cma 4.5.0, PySCIPOpt 6.3 and cvxpy 1.9. Set
`OPENBLAS_NUM_THREADS=1` when running several jobs in parallel.

## Reproducing the results

Every script is resumable: it appends one JSON line per finished run and skips finished runs when restarted.

```
A=ESDO,ESDO-R,PSO,DE,GA,GWO,HHO,L-SHADE,CMA-ES,WOA,SCA
python bench_center.py --dim 30 --runs 15 --algos $A --out results_p1/center.jsonl
python bench_origin.py --dim 30 --runs 10 --algos $A --out results_p1/origin.jsonl
python bench_port.py   --ports 1,2,3,4,5 --E 20 --runs 3 --algos $A --out results_p1/port_std.jsonl
python bench_mirror.py --mode mirror --ports 1,2,3,4,5 --E 20 --runs 3 --algos $A --out results_p1/port_mirror.jsonl
python bench_mirror.py --mode corner --ports 1,2,3,4,5 --E 20 --runs 3 --algos $A --out results_p1/port_corner.jsonl
python bench_mechanism.py --port 1 --lam 0.5 --runs 5 --algos $A --out results_p1/mechanism.jsonl
python bench_mechanism.py --port 2 --lam 0.5 --runs 5 --algos $A --out results_p1/mechanism.jsonl
python exact_port.py   --ports 1,2,3,4,5 --E 20 --tl 60 --out results_p1/port_exact.jsonl
python bench_port.py   --ports 1,2 --E 20 --runs 3 --budget 1000 --algos $A --out results_p1/port_std_b1000.jsonl
python bench_mirror.py --mode corner --ports 1,2 --E 20 --runs 3 --budget 1000 --algos $A --out results_p1/port_corner_b1000.jsonl
python exact_strong.py --cases 2:18,2:19,3:18,3:19,4:17,4:18,4:19,5:18 --time 1800 --out results_p1/port_exact_strong.jsonl
python analyze_p1.py
```

`exact_strong.py` starts each case from the best solution with stored weights found by the runs before it (direct
encoding, both budgets, and the 60 s model), so run it after the CCMV jobs. Its time limit is CPU time.

Seeds are fixed (portfolio runs: 100000 k + 1000 run + e for instance k; centre test: 500 + run; five-location test:
700 + run), so the direct, mirrored and randomly reflected runs are paired. The shifted optimum of the centre test is
drawn with `numpy.random.default_rng(2026)`, the random point and the off-diagonal corner of the five-location test with
`default_rng(2027)` and `default_rng(2028)`, and the reflected half of the coordinates in `--mode corner` with
`default_rng(77 + k)` for instance k.

## How to cite

If you use this code or these results, please cite the paper and this repository:

Abdelsalam, M., Diab, H., 2026. Code and results for: Location bias and encoding geometry in metaheuristic benchmarks
for cardinality-constrained portfolio optimization (version 1.0.0) [software]. GitHub.

The citation of the paper will be added here once it is published. GitHub also offers a citation from `CITATION.cff`
("Cite this repository").

## Licence

The code is released under the MIT License (see `LICENSE`). The OR-Library files in `data/orlib/` are by
J.E. Beasley and are included for convenience; see the OR-Library for their terms.
