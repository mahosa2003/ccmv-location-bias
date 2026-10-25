"""CEC 2017 bound-constrained single-objective test suite (Awad, Ali, Liang, Qu & Suganthan, 2016).

Python re-implementation of the official C code (cec17_test_func.c) using the official input data (shift vectors,
rotation matrices and shuffles in ../cec17_data). Numbering follows the official suite: F1 and F3-F30 (F2 was removed
from the competition), search range [-100, 100]^D, D in {2, 10, 20, 30, 50, 100}. make(fn, D) returns the error
f(x) - f*, with f* = 100 fn.

Behaviour of the reference code that is reproduced on purpose, so that values match published results:
  * F6 (Schaffer F7) and the Schaffer F7 parts of hybrids F14 and F20 read the global work vector y of the C code:
    in F6 this is the shifted but unrotated vector; inside a hybrid it is the first entries of the shuffled vector;
  * F8 (non-continuous Rastrigin): the rounding step is overwritten by the shift-and-rotate call, so F8 is the
    shifted and rotated Rastrigin function with its own data;
  * F9 (Levy): z = M (x - o) without the +1 shift, so the optimum is at z = 1 rather than at x = o (f* is still 900);
  * F20 (hybrid 10) is loaded through the composition-function branch, which uses the first shift row and matrix.
"""
import math
import os

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cec17_data')
FUNCS = [1] + list(range(3, 31))
PI, E = math.pi, math.e
NAMES = {1: 'Bent Cigar', 3: 'Zakharov', 4: 'Rosenbrock', 5: 'Rastrigin', 6: 'Expanded Schaffer F7',
         7: 'Lunacek bi-Rastrigin', 8: 'Non-continuous Rastrigin', 9: 'Levy', 10: 'Schwefel'}
NAMES.update({k: 'Hybrid %d' % (k - 10) for k in range(11, 21)})
NAMES.update({k: 'Composition %d' % (k - 20) for k in range(21, 31)})
GROUP = {k: ('Unimodal' if k <= 3 else 'Simple multimodal' if k <= 10 else 'Hybrid' if k <= 20 else 'Composition')
         for k in FUNCS}


# ------------------------------------------------------------------ shift / rotate (sr_func)
def _sr(x, o, M, rate, s, r):
    """Returns (z, y): z is the transformed vector, y the C code's global y after the call (None if untouched)."""
    if s:
        y = (x - o) * rate
        return (M @ y if r else y.copy()), y
    if r:
        y = x * rate
        return M @ y, y
    return x * rate, None


# ------------------------------------------------------------------ basic functions (signature of the C code)
def sphere(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 1.0, s, r)
    return float(np.sum(z * z))


def ellips(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 1.0, s, r)
    n = z.size
    return float(np.sum(10.0 ** (6.0 * np.arange(n) / (n - 1)) * z * z))


def bent_cigar(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 1.0, s, r)
    return float(z[0] * z[0] + 1e6 * np.sum(z[1:] * z[1:]))


def discus(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 1.0, s, r)
    return float(1e6 * z[0] * z[0] + np.sum(z[1:] * z[1:]))


def zakharov(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 1.0, s, r)
    s1 = np.sum(z * z)
    s2 = np.sum(0.5 * np.arange(1, z.size + 1) * z)
    return float(s1 + s2 ** 2 + s2 ** 4)


def levy(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 1.0, s, r)
    w = 1.0 + (z - 1.0) / 4.0
    t1 = math.sin(PI * w[0]) ** 2
    t3 = (w[-1] - 1) ** 2 * (1 + math.sin(2 * PI * w[-1]) ** 2)
    wi = w[:-1]
    return float(t1 + np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(PI * wi + 1) ** 2)) + t3)


def rosenbrock(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 2.048 / 100.0, s, r)
    z = z + 1.0
    a, b = z[:-1], z[1:]
    return float(np.sum(100.0 * (a * a - b) ** 2 + (a - 1.0) ** 2))


def schaffer_f7(x, o, M, s, r, yglob=None, **k):
    z, y = _sr(x, o, M, 1.0, s, r)
    n = x.size
    yv = y if y is not None else yglob[:n]          # the C code reads the global y, not z
    si = np.sqrt(yv[:-1] ** 2 + yv[1:] ** 2)
    t = np.sin(50.0 * si ** 0.2)
    f = np.sum(np.sqrt(si) + np.sqrt(si) * t * t)
    return float(f * f / (n - 1) / (n - 1))


def ackley(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 1.0, s, r)
    n = z.size
    s1 = -0.2 * math.sqrt(np.sum(z * z) / n)
    s2 = np.sum(np.cos(2.0 * PI * z)) / n
    return float(E - 20.0 * math.exp(s1) - math.exp(s2) + 20.0)


_WA = 0.5 ** np.arange(21)
_WB = 2.0 * PI * 3.0 ** np.arange(21)


def weierstrass(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 0.5 / 100.0, s, r)
    n = z.size
    s1 = np.sum(_WA[None, :] * np.cos(_WB[None, :] * (z[:, None] + 0.5)))
    s2 = np.sum(_WA * np.cos(_WB * 0.5))
    return float(s1 - n * s2)


def griewank(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 600.0 / 100.0, s, r)
    return float(1.0 + np.sum(z * z) / 4000.0 - np.prod(np.cos(z / np.sqrt(np.arange(1, z.size + 1)))))


def rastrigin(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 5.12 / 100.0, s, r)
    return float(np.sum(z * z - 10.0 * np.cos(2.0 * PI * z) + 10.0))


def schwefel(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 1000.0 / 100.0, s, r)
    n = z.size
    z = z + 4.209687462275036e+002
    f = 0.0
    hi, lo = z > 500, z < -500
    mid = ~(hi | lo)
    if hi.any():
        zm = np.fmod(z[hi], 500)
        f -= np.sum((500.0 - zm) * np.sin(np.sqrt(500.0 - zm)))
        f += np.sum(((z[hi] - 500.0) / 100) ** 2 / n)
    if lo.any():
        zm = np.fmod(np.abs(z[lo]), 500)
        f -= np.sum((-500.0 + zm) * np.sin(np.sqrt(500.0 - zm)))
        f += np.sum(((z[lo] + 500.0) / 100) ** 2 / n)
    zz = z[mid]
    f -= np.sum(zz * np.sin(np.sqrt(np.abs(zz))))
    return float(f + 4.189828872724338e+002 * n)


_K2 = 2.0 ** np.arange(1, 33)


def katsuura(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 5.0 / 100.0, s, r)
    n = z.size
    t = _K2[None, :] * z[:, None]
    temp = np.sum(np.abs(t - np.floor(t + 0.5)) / _K2[None, :], axis=1)
    f = float(np.prod((1.0 + np.arange(1, n + 1) * temp) ** (10.0 / n ** 1.2)))
    c = 10.0 / n / n
    return f * c - c


def bi_rastrigin(x, o, M, s, r, **k):
    n = x.size
    mu0, d = 2.5, 1.0
    sc = 1.0 - 1.0 / (2.0 * math.sqrt(n + 20.0) - 8.2)
    mu1 = -math.sqrt((mu0 * mu0 - d) / sc)
    y = (x - o[:n]) if s else x.copy()
    y = y * (10.0 / 100.0)
    tmpx = 2 * y
    tmpx = np.where(o[:n] < 0.0, -tmpx, tmpx)          # sign from the shift vector, as in the C code
    z = tmpx.copy()
    tmpx = tmpx + mu0
    t1 = np.sum((tmpx - mu0) ** 2)
    t2 = np.sum((tmpx - mu1) ** 2) * sc + d * n
    zz = M @ z if r else z
    return float(min(t1, t2) + 10.0 * (n - np.sum(np.cos(2.0 * PI * zz))))


def grie_rosen(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 5.0 / 100.0, s, r)
    z = z + 1.0
    a, b = z, np.roll(z, -1)                            # pairs (i, i+1) and the wrap (n-1, 0)
    t = 100.0 * (a * a - b) ** 2 + (a - 1.0) ** 2
    return float(np.sum(t * t / 4000.0 - np.cos(t) + 1.0))


def escaffer6(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 1.0, s, r)
    a, b = z, np.roll(z, -1)
    q = a * a + b * b
    return float(np.sum(0.5 + (np.sin(np.sqrt(q)) ** 2 - 0.5) / (1.0 + 0.001 * q) ** 2))


def happycat(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 5.0 / 100.0, s, r)
    n = z.size
    z = z - 1.0
    r2, sz = np.sum(z * z), np.sum(z)
    return float(abs(r2 - n) ** 0.25 + (0.5 * r2 + sz) / n + 0.5)


def hgbat(x, o, M, s, r, **k):
    z, _ = _sr(x, o, M, 5.0 / 100.0, s, r)
    n = z.size
    z = z - 1.0
    r2, sz = np.sum(z * z), np.sum(z)
    return float(abs(r2 ** 2 - sz ** 2) ** 0.5 + (0.5 * r2 + sz) / n + 0.5)


# ------------------------------------------------------------------ hybrid functions
_HF = {
    1: ([0.2, 0.4, 0.4], [zakharov, rosenbrock, rastrigin]),
    2: ([0.3, 0.3, 0.4], [ellips, schwefel, bent_cigar]),
    3: ([0.3, 0.3, 0.4], [bent_cigar, rosenbrock, bi_rastrigin]),
    4: ([0.2, 0.2, 0.2, 0.4], [ellips, ackley, schaffer_f7, rastrigin]),
    5: ([0.2, 0.2, 0.3, 0.3], [bent_cigar, hgbat, rastrigin, rosenbrock]),
    6: ([0.2, 0.2, 0.3, 0.3], [escaffer6, hgbat, rosenbrock, schwefel]),
    7: ([0.1, 0.2, 0.2, 0.2, 0.3], [katsuura, ackley, grie_rosen, schwefel, rastrigin]),
    8: ([0.2, 0.2, 0.2, 0.2, 0.2], [ellips, ackley, rastrigin, hgbat, discus]),
    9: ([0.2, 0.2, 0.2, 0.2, 0.2], [bent_cigar, rastrigin, grie_rosen, weierstrass, escaffer6]),
    10: ([0.1, 0.1, 0.2, 0.2, 0.2, 0.2], [hgbat, katsuura, ackley, rastrigin, schwefel, schaffer_f7]),
}


def hybrid(k, x, o, M, S, s=1, r=1):
    Gp, funcs = _HF[k]
    n = x.size
    G_nx = [int(math.ceil(g * n)) for g in Gp[:-1]]
    G_nx.append(n - sum(G_nx))
    z, _ = _sr(x, o, M, 1.0, s, r)
    y = z[S - 1]
    f, st = 0.0, 0
    for g, fun in zip(G_nx, funcs):
        f += fun(y[st:st + g], o, M, 0, 0, yglob=y)
        st += g
    return f


# ------------------------------------------------------------------ composition functions
def _cf_cal(x, O, delta, bias, fit):
    n = x.size
    fit = np.asarray(fit, float) + bias
    d2 = np.sum((x[None, :] - O[:len(fit)]) ** 2, axis=1)
    w = np.empty(len(fit))
    for i, v in enumerate(d2):
        w[i] = (1.0 / math.sqrt(v)) * math.exp(-v / 2.0 / n / delta[i] ** 2) if v != 0 else 1e99
    if w.max() == 0:
        w[:] = 1.0
    return float(np.sum(w / np.sum(w) * fit))


_CF = {  # (component, post-scaling factor applied to its value), delta, bias
    1: ([(rosenbrock, 1), (ellips, 1e4 / 1e10), (rastrigin, 1)], [10, 20, 30]),
    2: ([(rastrigin, 1), (griewank, 1e3 / 100), (schwefel, 1)], [10, 20, 30]),
    3: ([(rosenbrock, 1), (ackley, 1e3 / 100), (schwefel, 1), (rastrigin, 1)], [10, 20, 30, 40]),
    4: ([(ackley, 1e3 / 100), (ellips, 1e4 / 1e10), (griewank, 1e3 / 100), (rastrigin, 1)], [10, 20, 30, 40]),
    5: ([(rastrigin, 1e4 / 1e3), (happycat, 1e3 / 1e3), (ackley, 1e3 / 100), (discus, 1e4 / 1e10), (rosenbrock, 1)],
        [10, 20, 30, 40, 50]),
    6: ([(escaffer6, 1e4 / 2e7), (schwefel, 1), (griewank, 1e3 / 100), (rosenbrock, 1), (rastrigin, 1e4 / 1e3)],
        [10, 20, 20, 30, 40]),
    7: ([(hgbat, 1e4 / 1e3), (rastrigin, 1e4 / 1e3), (schwefel, 1e4 / 4e3), (bent_cigar, 1e4 / 1e30),
         (ellips, 1e4 / 1e10), (escaffer6, 1e4 / 2e7)], [10, 20, 30, 40, 50, 60]),
    8: ([(ackley, 1e3 / 100), (griewank, 1e3 / 100), (discus, 1e4 / 1e10), (rosenbrock, 1), (happycat, 1e3 / 1e3),
         (escaffer6, 1e4 / 2e7)], [10, 20, 30, 40, 50, 60]),
}
_CFH = {9: ([5, 6, 7], [10, 30, 50]), 10: ([5, 8, 9], [10, 30, 50])}


def _cf_scale(fun, v, factor):
    # the C code computes e.g. fit = 10000 * fit / 1e10; reproduce the operation order for exact values
    return v if factor == 1 else v * factor


def composition(k, x, O, Ms, SS=None):
    n = x.size
    if k in _CF:
        comps, delta = _CF[k]
        fit = []
        for i, (fun, fac) in enumerate(comps):
            fit.append(_cf_scale(fun, fun(x, O[i], Ms[i], 1, 1), fac))
        bias = 100.0 * np.arange(len(comps))
    else:
        hs, delta = _CFH[k]
        fit = [hybrid(h, x, O[i], Ms[i], SS[i * n:(i + 1) * n], 1, 1) for i, h in enumerate(hs)]
        bias = 100.0 * np.arange(len(hs))
    return _cf_cal(x, O, np.asarray(delta, float), bias, fit)


# ------------------------------------------------------------------ loader and public interface
def _load(fn, n):
    sh = np.loadtxt(os.path.join(_DATA, 'shift_data_%d.txt' % fn))
    Mraw = np.loadtxt(os.path.join(_DATA, 'M_%d_D%d.txt' % (fn, n)))
    SS = None
    if 11 <= fn <= 20 or fn in (29, 30):
        SS = np.loadtxt(os.path.join(_DATA, 'shuffle_data_%d_D%d.txt' % (fn, n))).astype(int).ravel()
    if fn < 20:
        o = sh.ravel()[:n]
        M = Mraw.ravel()[:n * n].reshape(n, n)
        return o, M, SS
    if sh.ndim == 1:
        sh = sh[None, :]
    O = sh[:, :n]
    Ms = Mraw.ravel()[: (Mraw.size // (n * n)) * n * n].reshape(-1, n, n)
    if fn == 20:                                         # hybrid loaded through the composition branch
        return O[0], Ms[0], SS
    return O, Ms, SS


def make(fn, n):
    """Error function f(x) - 100 fn for official CEC 2017 function fn (1, 3..30) in dimension n."""
    if fn == 2 or not 1 <= fn <= 30:
        raise ValueError('CEC 2017 functions are F1 and F3-F30')
    A, B, SS = _load(fn, n)
    basic = {1: bent_cigar, 3: zakharov, 4: rosenbrock, 5: rastrigin, 6: schaffer_f7, 7: bi_rastrigin,
             8: rastrigin, 9: levy, 10: schwefel}
    if fn in basic:
        fun = basic[fn]
        return lambda x: fun(np.asarray(x, float), A, B, 1, 1)
    if fn <= 20:
        return lambda x: hybrid(fn - 10, np.asarray(x, float), A, B, SS, 1, 1)
    return lambda x: composition(fn - 20, np.asarray(x, float), A, B, SS)


def optimum(fn, n):
    """The shift point (o or o_1). For F9 the true optimum is at z = 1, not at o."""
    A, B, SS = _load(fn, n)
    return A[0] if (fn > 20) else A
