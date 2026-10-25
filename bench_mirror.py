"""Control experiment for the portfolio encoding: the same CCMV problems with the mirrored encoding x -> 1 - x.

The decoding is scale-invariant (x and c x give the same portfolio), so with the standard encoding every optimum
touches the origin, which favours optimizers that drift towards the origin. Mirroring moves that apex to the corner
(1, ..., 1) without changing the problem. Resumable JSONL output.
Usage: python bench_mirror.py --ports 1 --E 20 --runs 1 --out results/port_mirror.jsonl
"""
import argparse, json, os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from esdo.portfolio import load_orlib, CCMV, lambdas  # noqa: E402
from esdo.algos import run_full, ALGOS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ports', default='1')
    ap.add_argument('--E', type=int, default=20)
    ap.add_argument('--budget', type=int, default=0,
                    help='evaluations per run as a multiple of n (0: the default max(10000, 100 n))')
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--algos', default=','.join(ALGOS))
    ap.add_argument('--out', default='results/port_mirror.jsonl')
    ap.add_argument('--mode', default='mirror', choices=['mirror', 'corner'],
                    help='mirror: x -> 1 - x for every coordinate; corner: a fixed random half of the coordinates')
    a = ap.parse_args()
    lam = lambdas(a.E)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); done.add((r['port'], r['alg'], r['run'], r['e']))
            except Exception:
                pass
    for r in range(a.runs):
        for k in [int(v) for v in a.ports.split(',')]:
            mu, S = load_orlib(k); m = CCMV(mu, S)
            flip = np.ones(m.n, bool) if a.mode == 'mirror' else np.random.default_rng(77 + k).random(m.n) < 0.5
            tf = lambda x, flip=flip: np.where(flip, 1.0 - np.asarray(x, float), np.asarray(x, float))
            for alg in a.algos.split(','):
                for e in range(a.E):
                    if (k, alg, r, e) in done:
                        continue
                    base = m.objective(lam[e])
                    f = lambda x, base=base, tf=tf: base(tf(x))
                    fe = a.budget * m.n if a.budget else max(10000, 100 * m.n)
                    t0 = time.time()
                    res = run_full(alg, f, np.zeros(m.n), np.ones(m.n), fe, 100000 * k + 1000 * r + e)
                    w = m.decode(tf(res['returned_x']))
                    v, ret = m.risk_return(w)
                    rec = {'port': k, 'alg': alg, 'run': r, 'e': e, 'lam': float(lam[e]), 'fe': fe, 'mode': a.mode,
                           'obj': float(res['returned']), 'var': v, 'ret': ret, 'time': round(time.time() - t0, 2)}
                    with open(a.out, 'a') as fh:
                        fh.write(json.dumps(rec) + '\n')
    print('finished', flush=True)


if __name__ == '__main__':
    main()
