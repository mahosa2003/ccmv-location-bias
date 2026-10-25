"""Egyptian Stray Dog Optimization (ESDO): faithful port of the original MATLAB code and a revised version.

Original: port of function ESDO() in eom/Copy_2_of_eomesdo.m (2 Nov 2023), the version used for the microgrid
cost-management study (Diab & Abdelsalam, Inventions 2024). The port keeps the code's logic, including:
  * territories are D equal cells along the diagonal of the search box (cell i = [L + (i-1)d, L + i d] in every
    dimension, d = (U - L)/D);
  * playful step = territory start + N(0,1) x territory width; with probability 0.05 a jump L + N(0,1) x 0.5 (U - L)
    centred on the lower bound;
  * territory "shrink"/"expand" multiplies both territory bounds by 0.9 / 1.1 (this moves the territory towards /
    away from the origin, not around its centre);
  * the dogs' current positions and energies are never updated after initialization, so alpha, beta and omega are
    fixed by the initial energies; the social step moves from the initial position towards alpha's / beta's best;
  * scavenging noise is an absolute N(0, 0.05) around alpha's best;
  * the returned solution is alpha's personal best (alpha = the dog that was best at initialization).
Two MATLAB indexing slips are corrected so that this logic runs as intended on problems with many variables: positions
are vectors (the saved code builds n x n matrices by implicit expansion and evaluates their first column), and the
territory width is end - start (the saved code uses territory(1) and territory(2), the starts of variables 1 and 2).
Candidates are clipped to the bounds (the MATLAB code has no bound handling). Everything else is unchanged; the
algorithm is scored on the value it returns.

Revised (ESDO-R): the six behaviours of the paper in the order of its pseudo-code (playful pass, re-ranking of the
pack, social pass), with the same parameters (D = 50 dogs, 0.95 / 0.05 playful split, alpha or beta with probability
0.5, 10 % of the pack resting every 5 iterations, scavenging probability 0.1), and these corrections:
  * dogs start uniformly in the box; each territory is a box centred on the dog's spot, half-width 0.1 (U - L);
  * playful step uniform inside the territory; with probability 0.05 a jump of up to half the box from the dog
    (no drift towards the lower bound or the origin);
  * greedy acceptance: a dog moves only to a better spot, so its position is its best-known spot;
  * territory size follows the one-fifth success rule (Rechenberg 1973): x 1.1 after a successful move inside the
    territory, x 1.1^(-1/4) after a failed one (the defensive behaviour: a dog that finds nothing better holds its
    spot and narrows its search around it);
  * alpha and beta are re-ranked every iteration; resting dogs make no move in that iteration;
  * scavenging: a dog behind the alpha samples the alpha's territory (relative scale instead of an absolute 0.05);
  * the best spot of the whole pack is returned; candidates are clipped to the bounds.
"""
import numpy as np


class Budget(Exception):
    pass


class Counted:
    """Objective wrapper: counts evaluations, stops at the budget and records the best value found."""

    def __init__(self, f, max_fe):
        self.f, self.max_fe, self.n, self.best, self.best_x = f, int(max_fe), 0, np.inf, None
        self.trace = []

    def __call__(self, x):
        if self.n >= self.max_fe:
            raise Budget()
        v = float(self.f(x))
        self.n += 1
        if v < self.best:
            self.best, self.best_x = v, np.array(x, float)
        if self.n % 1000 == 0:
            self.trace.append((self.n, self.best))
        return v


def esdo_original(fobj, lb, ub, max_fe, D=50, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    n = lb.size
    clip = lambda x: np.clip(x, lb, ub)
    tsize = (ub - lb) / D
    pos, energy, best_pos, best_e, terr = [], [], [], [], []
    returned = [np.inf, None]
    try:
        for i in range(D):
            t0 = lb + i * tsize
            t1 = t0 + tsize
            x = t0 + rng.random(n) * (t1 - t0)
            e = fobj(x)
            pos.append(x); energy.append(e); best_pos.append(x.copy()); best_e.append(e); terr.append([t0.copy(), t1.copy()])
        order = np.argsort(energy)
        ia, ib = order[0], order[1]
        alpha_pos, alpha_e = best_pos[ia].copy(), best_e[ia]
        beta_pos = best_pos[ib].copy()
        returned = [alpha_e, alpha_pos]
        it = 0
        while True:
            it += 1
            for i in range(D):
                if rng.random() < 0.95:
                    new = terr[i][0] + rng.standard_normal(n) * (terr[i][1] - terr[i][0])
                else:
                    new = lb + rng.standard_normal(n) * (ub - lb) * 0.5
                new = clip(new)
                e = fobj(new)
                if e < best_e[i]:
                    best_e[i], best_pos[i] = e, new
                    if e < alpha_e * 0.95:
                        terr[i] = [terr[i][0] * 0.9, terr[i][1] * 0.9]
                if rng.random() < 0.5:
                    new = pos[i] + rng.random() * (alpha_pos - pos[i])
                else:
                    new = pos[i] + rng.random() * (beta_pos - pos[i])
                new = clip(new)
                e = fobj(new)
                if e < best_e[i]:
                    best_e[i], best_pos[i] = e, new
                    if i == D - 1:
                        terr[i] = [terr[i][0] * 1.1, terr[i][1] * 1.1]
                if it % 5 == 0 and (i + 1) <= D / 10:
                    continue
                if rng.random() < 0.1 and e > alpha_e:
                    new = clip(alpha_pos + rng.standard_normal(n) * 0.05)
                    e2 = fobj(new)
                    if e2 < best_e[i]:
                        best_e[i], best_pos[i] = e2, new
            # leaders re-read from the (never updated) initial energies: same dogs every iteration
            alpha_pos, alpha_e = best_pos[ia].copy(), best_e[ia]
            beta_pos = best_pos[ib].copy()
            returned = [alpha_e, alpha_pos]
    except Budget:
        pass
    return returned[0], returned[1]


def esdo_revised(fobj, lb, ub, max_fe, D=50, seed=None, w0=0.1):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    n = lb.size
    span = ub - lb
    clip = lambda x: np.clip(x, lb, ub)
    X = lb + rng.random((D, n)) * span       # positions (= best-known spots, greedy acceptance)
    F = np.full(D, np.inf)
    W = np.full(D, w0)                       # territory half-width as a share of the box
    grow, shrink = 1.1, 1.1 ** -0.25         # one-fifth success rule
    try:
        for i in range(D):
            F[i] = fobj(X[i])
        it = 0
        while True:
            it += 1
            # rest: every 5th iteration a random 10% of the pack sleeps (no moves, no evaluations)
            rest = set(rng.choice(D, size=max(1, D // 10), replace=False).tolist()) if it % 5 == 0 else set()
            # playful (territorial) pass
            for i in range(D):
                if i in rest:
                    continue
                if rng.random() < 0.95:              # inside the territory
                    cand = clip(X[i] + (2 * rng.random(n) - 1) * W[i] * span)
                    e = fobj(cand)
                    if e < F[i]:
                        X[i], F[i] = cand, e
                        W[i] = min(W[i] * grow, 0.5)
                    else:                            # defensive: hold the spot, narrow the territory
                        W[i] = max(W[i] * shrink, 1e-15)
                else:                                # occasional jump of up to half the box
                    cand = clip(X[i] + (2 * rng.random(n) - 1) * 0.5 * span)
                    e = fobj(cand)
                    if e < F[i]:
                        X[i], F[i] = cand, e
            # social hierarchy: re-rank the pack
            order = np.argsort(F)
            ia, ib = order[0], order[1]
            alpha, beta, f_alpha = X[ia].copy(), X[ib].copy(), F[ia]
            for i in range(D):
                if i in rest:
                    continue
                lead = alpha if rng.random() < 0.5 else beta
                cand = clip(X[i] + rng.random() * (lead - X[i]))
                e = fobj(cand)
                if e < F[i]:
                    X[i], F[i] = cand, e
                # scavenging: a dog behind the alpha searches the alpha's territory
                if F[i] > f_alpha and rng.random() < 0.1:
                    cand = clip(alpha + (2 * rng.random(n) - 1) * W[ia] * span)
                    e = fobj(cand)
                    if e < F[i]:
                        X[i], F[i] = cand, e
    except Budget:
        pass
    i = int(np.argmin(F))
    return F[i], X[i].copy()
