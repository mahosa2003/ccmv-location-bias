"""Exact (or best-bound) CCMV solutions with SCIP for every OR-Library instance and lambda of the benchmark.

Writes results/port_exact.jsonl (resumable). A time limit per lambda keeps the run bounded; the status and the dual
bound are stored so that proven optima can be told apart from incumbents.
Usage: python exact_port.py --ports 1,2,3,4,5 --E 20 --tl 60
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esdo.portfolio import load_orlib, CCMV, lambdas, exact_ccmv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ports', default='1,2,3,4,5')
    ap.add_argument('--E', type=int, default=20)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--tl', type=float, default=60.0)
    ap.add_argument('--out', default='results/port_exact.jsonl')
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r['port'], r['e']))
    lam = lambdas(a.E)
    for k in [int(v) for v in a.ports.split(',')]:
        mu, S = load_orlib(k)
        m = CCMV(mu, S, K=a.K)
        for e in range(a.E):
            if (k, e) in done:
                continue
            r = exact_ccmv(m, float(lam[e]), time_limit=a.tl)
            w = np.where(r['w'] > 1e-9, r['w'], 0.0)
            v, ret = m.risk_return(w)
            held = np.flatnonzero(w > 0)
            rec = {'port': k, 'e': e, 'lam': float(lam[e]), 'status': r['status'], 'obj': r['obj'],
                   'bound': r['bound'], 'var': v, 'ret': ret, 'assets': held.tolist(),
                   'weights': [round(float(x), 8) for x in w[held]], 'time': round(r['time'], 2)}
            with open(a.out, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
            print(k, e, r['status'], '%.6e' % r['obj'], '%.1fs' % r['time'], flush=True)


if __name__ == '__main__':
    main()
