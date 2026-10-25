"""Mechanism check for the encoding effect: where in the box does each optimizer search?

For one CCMV instance and one risk-aversion value, every evaluated point x is logged in blocks of 250 evaluations:
the mean coordinate of the block (0 = origin corner, 1 = opposite corner) and the best objective so far. Run with the
standard encoding and with the mirrored encoding x -> 1 - x. Resumable JSONL output.
Usage: python bench_mechanism.py --port 1 --lam 0.5 --runs 5 --out results_p1/mechanism.jsonl
"""
import argparse, json, os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from esdo.portfolio import load_orlib, CCMV  # noqa: E402
from esdo.algos import run_full, ALGOS_P1  # noqa: E402


class Logger:
    def __init__(self, f, block=250):
        self.f, self.block = f, block
        self.n, self.acc, self.best, self.trace = 0, 0.0, np.inf, []

    def __call__(self, x):
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        v = self.f(x)
        self.n += 1
        self.acc += float(x.mean())
        self.best = min(self.best, v)
        if self.n % self.block == 0:
            self.trace.append([self.n, round(self.acc / self.block, 5), self.best])
            self.acc = 0.0
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=1)
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--algos', default=','.join(ALGOS_P1))
    ap.add_argument('--out', default='results_p1/mechanism.jsonl')
    a = ap.parse_args()
    mu, S = load_orlib(a.port)
    m = CCMV(mu, S)
    base = m.objective(a.lam)
    fe = max(10000, 100 * m.n)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); done.add((r['port'], r['lam'], r['alg'], r['enc'], r['run']))
            except Exception:
                pass
    for run in range(a.runs):
        for alg in a.algos.split(','):
            for enc in ['std', 'mirror']:
                if (a.port, a.lam, alg, enc, run) in done:
                    continue
                g = base if enc == 'std' else (lambda x: base(1.0 - np.asarray(x, float)))
                log = Logger(g)
                t0 = time.time()
                res = run_full(alg, log, np.zeros(m.n), np.ones(m.n), fe, 9000 + run)
                xr = np.asarray(res['returned_x'], float)
                rec = {'port': a.port, 'lam': a.lam, 'alg': alg, 'enc': enc, 'run': run, 'fe': fe,
                       'obj': float(res['returned']), 'final_mean_x': float(np.clip(xr, 0, 1).mean()),
                       'final_norm_x': float(np.linalg.norm(np.clip(xr, 0, 1))), 'trace': log.trace,
                       'time': round(time.time() - t0, 2)}
                with open(a.out, 'a') as fh:
                    fh.write(json.dumps(rec) + '\n')
    print('finished', flush=True)


if __name__ == '__main__':
    main()
