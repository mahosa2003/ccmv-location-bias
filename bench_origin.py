"""Origin bias versus centre bias. The search box is [0, 200]^D, so the origin is a corner and the centre is 100.
Each function is solved with its optimum at the origin, at the opposite corner (200, ..., 200), at a corner off the
main diagonal (a fixed random vertex), at the centre and at a fixed random point in [20, 180]^D. An origin-biased
optimizer does best at the origin but not at the opposite corner; one whose moves are coupled along the main diagonal
does well at both diagonal corners but not at the off-diagonal one; one drawn to the bounds (for example by clipping)
does well at every corner; a centre-biased one does best at the centre; a shift-invariant one does alike everywhere.
Usage: python bench_origin.py --dim 30 --runs 10 --out results_p1/origin.jsonl
"""
import argparse, json, os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from esdo.algos import run, ALGOS_P1  # noqa: E402


def base_functions():
    def sphere(z): return float(np.sum(z * z))
    def rastrigin(z):
        z = 0.0512 * z; return float(np.sum(z * z - 10 * np.cos(2 * np.pi * z) + 10))
    def ackley(z):
        z = 0.32 * z
        return float(-20 * np.exp(-0.2 * np.sqrt(np.mean(z * z))) - np.exp(np.mean(np.cos(2 * np.pi * z))) + 20 + np.e)
    def griewank(z):
        z = 6.0 * z; return float(1 + np.sum(z * z) / 4000 - np.prod(np.cos(z / np.sqrt(np.arange(1, z.size + 1)))))
    return [('Sphere', sphere), ('Rastrigin', rastrigin), ('Ackley', ackley), ('Griewank', griewank)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=30)
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--fe', type=int, default=0)
    ap.add_argument('--algos', default=','.join(ALGOS_P1))
    ap.add_argument('--out', default='results_p1/origin.jsonl')
    a = ap.parse_args()
    D = a.dim; FE = a.fe or 2000 * D
    lb, ub = np.zeros(D), 200.0 * np.ones(D)
    opt = {'origin': np.zeros(D), 'centre': 100.0 * np.ones(D),
           'random': np.random.default_rng(2027).uniform(20, 180, D), 'corner': 200.0 * np.ones(D),
           'offcorner': 200.0 * (np.random.default_rng(2028).random(D) < 0.5)}
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); done.add((r['func'], r['case'], r['alg'], r['run']))
            except Exception:
                pass
    tasks = [(fn, g, case, alg, k) for k in range(a.runs) for fn, g in base_functions() for case in opt
             for alg in a.algos.split(',')]
    for fn, g, case, alg, k in tasks:
        if (fn, case, alg, k) in done:
            continue
        o = opt[case]
        f = (lambda g, o: (lambda x: g(np.asarray(x) - o)))(g, o)
        t0 = time.time()
        best, returned, nfe, trace = run(alg, f, lb, ub, FE, 700 + k)
        err = returned if alg == 'ESDO' else best
        rec = {'func': fn, 'case': case, 'alg': alg, 'run': k, 'D': D, 'FE': FE, 'err': float(err),
               'best_seen': float(best), 'time': round(time.time() - t0, 2)}
        with open(a.out, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    print('finished', flush=True)


if __name__ == '__main__':
    main()
