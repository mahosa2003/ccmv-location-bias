"""CCMV portfolio benchmark on the five OR-Library instances (Chang et al., 2000).

For each instance, algorithm, run and risk-aversion weight lam (E equally spaced values in [0, 1]) the optimizer gets
max(10000, 100 n) evaluations of the encoded objective. Resumable JSONL output, one line per task.
Usage: python bench_port.py --ports 1,2,3,4,5 --E 20 --runs 3 --out results/port.jsonl --worker 0 --nworkers 2
"""
import argparse, json, os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from esdo.portfolio import load_orlib, CCMV, lambdas  # noqa: E402
from esdo.algos import run_full, ALGOS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ports', default='1,2,3,4,5')
    ap.add_argument('--E', type=int, default=20)
    ap.add_argument('--budget', type=int, default=0,
                    help='evaluations per run as a multiple of n (0: the default max(10000, 100 n))')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--algos', default=','.join(ALGOS))
    ap.add_argument('--out', default='results/port.jsonl')
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--nworkers', type=int, default=1)
    a = ap.parse_args()
    ports = [int(v) for v in a.ports.split(',')]
    algos = a.algos.split(',')
    lam = lambdas(a.E)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); done.add((r['port'], r['alg'], r['run'], r['e']))
            except Exception:
                pass
    tasks = [(k, alg, r, e) for r in range(a.runs) for k in ports for alg in algos for e in range(a.E)]
    tasks = [t for i, t in enumerate(tasks) if i % a.nworkers == a.worker and t not in done]
    print('worker %d: %d tasks' % (a.worker, len(tasks)), flush=True)
    cache = {}
    for k, alg, r, e in tasks:
        if k not in cache:
            mu, S = load_orlib(k); cache[k] = CCMV(mu, S, K=a.K)
        m = cache[k]
        fe = a.budget * m.n if a.budget else max(10000, 100 * m.n)
        t0 = time.time()
        res = run_full(alg, m.objective(lam[e]), np.zeros(m.n), np.ones(m.n), fe, 100000 * k + 1000 * r + e)
        w = m.decode(res['returned_x'])
        v, ret = m.risk_return(w)
        held = np.flatnonzero(w > 0)
        rec = {'port': k, 'alg': alg, 'run': r, 'e': e, 'lam': float(lam[e]), 'fe': fe, 'nfe': int(res['nfe']),
               'obj': float(res['returned']), 'obj_best_seen': float(res['best']), 'var': v, 'ret': ret,
               'assets': held.tolist(), 'weights': [round(float(x), 8) for x in w[held]],
               'time': round(time.time() - t0, 2)}
        with open(a.out, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    print('worker %d finished' % a.worker, flush=True)


if __name__ == '__main__':
    main()
