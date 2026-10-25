"""Centre-bias test: the same functions with the optimum at the centre of the box (x* = 0) and shifted to a random
point o ~ U(-80, 80)^D. Shift-invariant optimizers perform alike on both; centre-biased ones do much better at 0.
Usage: python bench_center.py --dim 30 --runs 5 --out results/center.jsonl --worker 0 --nworkers 2
"""
import argparse, json, os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from esdo.algos import run, ALGOS  # noqa: E402


def funcs(D):
    o = np.random.default_rng(2026).uniform(-80, 80, D)
    def sphere(z): return float(np.sum(z * z))
    def rastrigin(z):
        z = 0.0512 * z; return float(np.sum(z * z - 10 * np.cos(2 * np.pi * z) + 10))
    def ackley(z):
        z = 0.32 * z
        return float(-20 * np.exp(-0.2 * np.sqrt(np.mean(z * z))) - np.exp(np.mean(np.cos(2 * np.pi * z))) + 20 + np.e)
    def griewank(z):
        z = 6.0 * z; return float(1 + np.sum(z * z) / 4000 - np.prod(np.cos(z / np.sqrt(np.arange(1, z.size + 1)))))
    out = {}
    for name, g in [('Sphere', sphere), ('Rastrigin', rastrigin), ('Ackley', ackley), ('Griewank', griewank)]:
        out[(name, 'centre')] = (lambda g: (lambda x: g(np.asarray(x))))(g)
        out[(name, 'shifted')] = (lambda g: (lambda x: g(np.asarray(x) - o)))(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=30)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--fe', type=int, default=0)
    ap.add_argument('--algos', default=','.join(ALGOS))
    ap.add_argument('--out', default='results/center.jsonl')
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--nworkers', type=int, default=1)
    a = ap.parse_args()
    D = a.dim; FE = a.fe or 2000 * D
    F = funcs(D)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); done.add((r['func'], r['case'], r['alg'], r['run']))
            except Exception:
                pass
    tasks = [(fn, case, alg, k) for k in range(a.runs) for (fn, case) in F for alg in a.algos.split(',')]
    tasks = [t for i, t in enumerate(tasks) if i % a.nworkers == a.worker and t not in done]
    lb, ub = -100.0 * np.ones(D), 100.0 * np.ones(D)
    print('worker %d: %d tasks' % (a.worker, len(tasks)), flush=True)
    for fn, case, alg, k in tasks:
        t0 = time.time()
        best, returned, nfe, trace = run(alg, F[(fn, case)], lb, ub, FE, 500 + k)
        err = returned if alg == 'ESDO' else best
        rec = {'func': fn, 'case': case, 'alg': alg, 'run': k, 'D': D, 'FE': FE, 'err': float(err),
               'best_seen': float(best), 'time': round(time.time() - t0, 2)}
        with open(a.out, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    print('worker %d finished' % a.worker, flush=True)


if __name__ == '__main__':
    main()
