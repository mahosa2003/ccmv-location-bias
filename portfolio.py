"""Cardinality-constrained mean-variance (CCMV) portfolio model of Chang, Meade, Beasley & Sharaiha (2000).

    min  lam * w' S w - (1 - lam) * mu' w
    s.t. sum(w) = 1,  exactly K assets held,  eps <= w_i <= delta for held assets.

Continuous encoding for metaheuristics: x in [0, 1]^n; the K assets with the largest x_i are held and weighted
w_i = eps + (1 - K eps) x_i / sum_{held} x_j (equal shares if all held x_j are 0). With delta = 1 the upper bound is
never active (w_i <= 1 - (K - 1) eps). Every x therefore maps to a feasible portfolio.
"""
import os

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'orlib')
MARKETS = {1: 'Hang Seng', 2: 'DAX 100', 3: 'FTSE 100', 4: 'S&P 100', 5: 'Nikkei 225'}


def load_orlib(k):
    """OR-Library portk.txt: weekly mean returns, standard deviations and correlations (Chang et al., 2000)."""
    tok = open(os.path.join(_DATA, 'port%d.txt' % k)).read().split()
    n = int(tok[0])
    a = np.array(tok[1:1 + 2 * n], float).reshape(n, 2)
    mu, sd = a[:, 0], a[:, 1]
    rest = np.array(tok[1 + 2 * n:], float).reshape(-1, 3)
    R = np.eye(n)
    i, j = rest[:, 0].astype(int) - 1, rest[:, 1].astype(int) - 1
    R[i, j] = rest[:, 2]
    R[j, i] = rest[:, 2]
    S = R * np.outer(sd, sd)
    return mu, S


def load_uef(k):
    """Unconstrained efficient frontier portefk.txt as (return, variance), sorted by return."""
    a = np.loadtxt(os.path.join(_DATA, 'portef%d.txt' % k))
    return a[np.argsort(a[:, 0])]


class CCMV:
    def __init__(self, mu, S, K=10, eps=0.01, delta=1.0):
        self.mu, self.S, self.K, self.eps, self.delta = np.asarray(mu, float), np.asarray(S, float), K, eps, delta
        self.n = self.mu.size

    def decode(self, x):
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        idx = np.argpartition(-x, self.K - 1)[:self.K]
        s = x[idx].sum()
        share = x[idx] / s if s > 0 else np.full(self.K, 1.0 / self.K)
        w = np.zeros(self.n)
        w[idx] = self.eps + (1.0 - self.K * self.eps) * share
        return w

    def risk_return(self, w):
        return float(w @ self.S @ w), float(self.mu @ w)

    def objective(self, lam):
        mu, S, K, eps = self.mu, self.S, self.K, self.eps

        def f(x):
            # same mapping as decode(), evaluated on the K held assets only
            x = np.clip(np.asarray(x, float), 0.0, 1.0)
            idx = np.argpartition(-x, K - 1)[:K]
            s = x[idx].sum()
            share = x[idx] / s if s > 0 else np.full(K, 1.0 / K)
            ws = eps + (1.0 - K * eps) * share
            return lam * float(ws @ S[np.ix_(idx, idx)] @ ws) - (1.0 - lam) * float(mu[idx] @ ws)
        return f


def lambdas(E):
    return np.linspace(0.0, 1.0, E)


def percentage_errors(points, uef):
    """Percentage deviation of portfolios from the unconstrained efficient frontier, as defined by Chang et al. (2000),
    Section 5.2.2: frontier and portfolios are taken in (standard deviation, return) coordinates, the frontier is
    linearly interpolated between its listed points, and each (variance, return) point gets the smaller of
    |100 (s - s**) / s**|, with s** the frontier standard deviation at the same return, and |100 (r - r**) / r**|,
    with r** the frontier return at the same standard deviation. A direction is skipped when the point lies outside
    the range of the listed frontier."""
    r_u, s_u = uef[:, 0], np.sqrt(uef[:, 1])
    order_s = np.argsort(s_u)
    out = []
    for v, r in points:
        s = np.sqrt(v)
        errs = []
        if r_u[0] <= r <= r_u[-1]:
            ss = np.interp(r, r_u, s_u)
            errs.append(100.0 * abs(s - ss) / ss)
        if s_u[order_s[0]] <= s <= s_u[order_s[-1]]:
            rs = np.interp(s, s_u[order_s], r_u[order_s])
            errs.append(100.0 * abs(r - rs) / rs)
        out.append(min(errs) if errs else np.nan)
    return np.array(out)


def exact_ccmv(model, lam, time_limit=120.0, gap=1e-6):
    """Exact solution with SCIP (pyscipopt): convex MIQCP with an epigraph variable for the variance."""
    from pyscipopt import Model, quicksum
    n, K, eps, delta = model.n, model.K, model.eps, model.delta
    m = Model()
    m.hideOutput()
    m.setParam('limits/time', time_limit)
    m.setParam('limits/gap', gap)
    w = [m.addVar(lb=0.0, ub=delta, name='w%d' % i) for i in range(n)]
    z = [m.addVar(vtype='B', name='z%d' % i) for i in range(n)]
    t = m.addVar(lb=0.0, name='t')
    m.addCons(quicksum(w) == 1.0)
    m.addCons(quicksum(z) == K)
    for i in range(n):
        m.addCons(w[i] >= eps * z[i])
        m.addCons(w[i] <= delta * z[i])
    S = model.S
    m.addCons(quicksum(S[i, j] * w[i] * w[j] for i in range(n) for j in range(n)) <= t)
    m.setObjective(lam * t - (1.0 - lam) * quicksum(model.mu[i] * w[i] for i in range(n)), 'minimize')
    m.optimize()
    status = m.getStatus()
    if m.getNSols() == 0:
        return None
    wv = np.array([m.getVal(v) for v in w])
    return {'status': status, 'obj': float(m.getObjVal()), 'bound': float(m.getDualbound()), 'w': wv,
            'time': float(m.getSolvingTime())}
