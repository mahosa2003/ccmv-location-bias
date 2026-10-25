"""Exact CCMV with a strengthened formulation, for the frontier points that the plain model did not close.

The covariance is split as Sigma = (Sigma - D) + D with a diagonal D >= 0 such that Sigma - D is positive semidefinite
(D maximizes trace(D), an SDP solved with cvxpy). The diagonal part is written in perspective form, w_i^2 <= s_i z_i,
which tightens the continuous relaxation of the cardinality and minimum-weight constraints (Frangioni and Gentile,
2006; Zheng, Sun and Li, 2014). The best known solution is passed to SCIP as a starting incumbent.
Usage: python exact_strong.py --cases 2:19,4:19 --time 1800 --out results_p1/port_exact_strong.jsonl
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esdo.portfolio import load_orlib, CCMV, lambdas  # noqa: E402


def max_diag(S):
    import cvxpy as cp
    n = S.shape[0]
    d = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Maximize(cp.sum(d)), [S - cp.diag(d) >> 0])
    if n <= 150:
        prob.solve(solver='CLARABEL')
    else:   # Clarabel needs about 5 GB for the 225-asset instance; the first-order SCS needs about 150 MB
        prob.solve(solver='SCS', eps=1e-8, max_iters=200000)
    dv = np.maximum(np.asarray(d.value).ravel(), 0.0) * (1 - 1e-6)
    # make sure the remainder is PSD after rounding
    ev = np.linalg.eigvalsh(S - np.diag(dv)).min()
    if ev < 0:
        dv = np.maximum(dv + ev * 1.01, 0.0)
    return dv


def solve(m, lam, D, start_w, time_limit):
    from pyscipopt import Model, quicksum
    n, K, eps, delta = m.n, m.K, m.eps, m.delta
    S = m.S
    R = S - np.diag(D)
    M = Model()
    M.hideOutput()
    M.setParam('timing/clocktype', 1)                  # CPU time, so that a low-priority run gets its full limit
    M.setParam('limits/time', time_limit)
    M.setParam('limits/gap', 1e-6)
    w = [M.addVar(lb=0.0, ub=delta, name='w%d' % i) for i in range(n)]
    z = [M.addVar(vtype='B', name='z%d' % i) for i in range(n)]
    s = [M.addVar(lb=0.0, ub=delta * delta, name='s%d' % i) for i in range(n)]
    t = M.addVar(lb=0.0, name='t')
    M.addCons(quicksum(w) == 1.0)
    M.addCons(quicksum(z) == K)
    for i in range(n):
        M.addCons(w[i] >= eps * z[i])
        M.addCons(w[i] <= delta * z[i])
        M.addCons(w[i] * w[i] <= s[i] * z[i])            # perspective of d_i w_i^2 (rotated cone)
        M.addCons(s[i] <= delta * delta * z[i])
    M.addCons(quicksum(R[i, j] * w[i] * w[j] for i in range(n) for j in range(n)) + quicksum(D[i] * s[i] for i in range(n)) <= t)
    M.setObjective(lam * t - (1.0 - lam) * quicksum(m.mu[i] * w[i] for i in range(n)), 'minimize')
    if start_w is not None:
        sol = M.createSol()
        held = start_w > 1e-12
        for i in range(n):
            M.setSolVal(sol, w[i], float(start_w[i]))
            M.setSolVal(sol, z[i], 1.0 if held[i] else 0.0)
            M.setSolVal(sol, s[i], float(start_w[i] ** 2))
        M.setSolVal(sol, t, float(start_w @ R @ start_w + D @ (start_w ** 2)))
        stored = bool(M.addSol(sol, free=True))
    else:
        stored = False
    t0 = time.time()
    M.optimize()
    wv = np.array([M.getVal(v) for v in w]) if M.getNSols() else None
    return {'status': M.getStatus(), 'obj': float(M.getObjVal()) if M.getNSols() else None, 'bound': float(M.getDualbound()),
            'w': wv, 'time': round(time.time() - t0, 2), 'nodes': int(M.getNNodes()), 'start_stored': stored}


def best_known(port, e):
    """Best solution with stored weights (direct-encoding runs at both budgets and the plain exact model)."""
    best = None
    for f in ['port_std.jsonl', 'port_std_b1000.jsonl', 'port_exact.jsonl']:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_p1', f)
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('port') == port and r.get('e') == e and r.get('obj') is not None and 'assets' in r and \
                    (best is None or r['obj'] < best['obj']):
                best = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True, help='port:e pairs, comma separated')
    ap.add_argument('--time', type=float, default=1800.0)
    ap.add_argument('--E', type=int, default=20)
    ap.add_argument('--out', default='results_p1/port_exact_strong.jsonl')
    a = ap.parse_args()
    lam = lambdas(a.E)
    diag_cache = {}
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); done.add((r['port'], r['e']))
            except Exception:
                pass
    for item in a.cases.split(','):
        port, e = [int(v) for v in item.split(':')]
        if (port, e) in done:
            continue
        mu, S = load_orlib(port)
        m = CCMV(mu, S)
        if port not in diag_cache:
            t0 = time.time()
            diag_cache[port] = max_diag(m.S)
            print('port%d: diagonal SDP %.1fs, trace(D)/trace(S) = %.3f' % (port, time.time() - t0, diag_cache[port].sum() / np.trace(m.S)), flush=True)
        bk = best_known(port, e)
        w0 = None
        if bk is not None and 'assets' in bk:
            w0 = np.zeros(m.n); w0[bk['assets']] = bk['weights']; w0 /= w0.sum()
        res = solve(m, float(lam[e]), diag_cache[port], w0, a.time)
        rec = {'port': port, 'e': e, 'lam': float(lam[e]), 'status': res['status'], 'obj': res['obj'], 'bound': res['bound'],
               'time': res['time'], 'nodes': res['nodes'], 'start_obj': bk['obj'] if bk else None,
               'start_stored': res['start_stored']}
        if res['w'] is not None:
            w = res['w']; held = np.where(w > 1e-9)[0]
            v, r = m.risk_return(w)
            rec.update({'var': v, 'ret': r, 'assets': held.tolist(), 'weights': [round(float(w[i]), 8) for i in held]})
        with open(a.out, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        print(rec['port'], rec['e'], rec['status'], rec['obj'], rec['bound'], rec['time'], flush=True)


if __name__ == '__main__':
    main()
