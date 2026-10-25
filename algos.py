"""Common interface to ESDO variants and baseline optimizers with exact evaluation budgets.

Every optimizer receives the same Counted objective wrapper, which stops the run at the budget; the value recorded
for each run is the best objective value evaluated within the budget (for ESDO-original also the value it returns).
"""
import warnings
import numpy as np

from .core import Counted, Budget, esdo_original, esdo_revised

warnings.filterwarnings('ignore')
ALGOS = ['ESDO', 'ESDO-R', 'PSO', 'DE', 'GA', 'GWO', 'HHO', 'L-SHADE', 'CMA-ES']
ALGOS_P1 = ALGOS + ['WOA', 'SCA']          # paper 1 adds two more widely used metaphor-based methods


def _mealpy(name, fc, lb, ub, max_fe, seed):
    from mealpy import FloatVar
    from mealpy.swarm_based import PSO, GWO, HHO, WOA
    from mealpy.math_based import SCA
    from mealpy.evolutionary_based import DE, GA, SHADE
    n = len(lb)
    # the epoch count drives the schedules of several optimizers (e.g. GWO's a, PSO's inertia, L-SHADE's population
    # size), so it is set to end when the budget is spent; the Counted wrapper enforces the exact budget
    # L-SHADE starts with 18 n individuals (Tanabe & Fukunaga, 2014), a size tuned for budgets of 10000 n; with
    # smaller budgets the initial population is capped at max_fe / 100 so that it still runs ~100 generations
    pop_ls = int(max(50, min(18 * n, max_fe // 100)))
    per_epoch = {'PSO': 50, 'DE': 50, 'GA': 50, 'GWO': 50, 'HHO': 75, 'L-SHADE': pop_ls, 'WOA': 50, 'SCA': 50}[name]
    ep = max(2, int(max_fe / per_epoch)) + 2
    if name == 'PSO':
        model = PSO.OriginalPSO(epoch=ep, pop_size=50, c1=2.05, c2=2.05, w=0.4)
    elif name == 'DE':
        model = DE.OriginalDE(epoch=ep, pop_size=50, wf=0.5, cr=0.9, strategy=0)
    elif name == 'GA':
        model = GA.BaseGA(epoch=ep, pop_size=50, pc=0.9, pm=0.05)
    elif name == 'GWO':
        model = GWO.OriginalGWO(epoch=ep, pop_size=50)
    elif name == 'HHO':
        model = HHO.OriginalHHO(epoch=ep, pop_size=50)
    elif name == 'WOA':
        model = WOA.OriginalWOA(epoch=ep, pop_size=50)
    elif name == 'SCA':
        model = SCA.OriginalSCA(epoch=ep, pop_size=50)
    elif name == 'L-SHADE':
        model = SHADE.L_SHADE(epoch=ep, pop_size=pop_ls, miu_f=0.5, miu_cr=0.5)
    else:
        raise ValueError(name)
    problem = {'obj_func': fc, 'bounds': FloatVar(lb=list(lb), ub=list(ub)), 'minmax': 'min', 'log_to': None}
    try:
        model.solve(problem, seed=seed)
    except Budget:
        pass


def _cma(fc, lb, ub, max_fe, seed):
    import cma
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    rng = np.random.default_rng(seed)
    x0 = lb + rng.random(lb.size) * (ub - lb)
    opts = {'bounds': [list(lb), list(ub)], 'maxfevals': max_fe, 'verbose': -9, 'seed': int(seed) + 1,
            'tolfun': 1e-14, 'tolx': 1e-14}
    try:
        cma.fmin2(fc, x0, 0.3 * float(np.mean(ub - lb)), opts, restarts=20, bipop=True)
    except Budget:
        pass


def run_full(name, f, lb, ub, max_fe, seed):
    """Runs one optimizer; returns a dict with the best value and point evaluated within the budget, the value and
    point the algorithm itself returns (they differ only for ESDO-original), evaluations used and the trace."""
    fc = Counted(f, max_fe)
    ret_v, ret_x = np.nan, None
    if name == 'ESDO':
        ret_v, ret_x = esdo_original(fc, lb, ub, max_fe, seed=seed)
    elif name == 'ESDO-R':
        ret_v, ret_x = esdo_revised(fc, lb, ub, max_fe, seed=seed)
    elif name == 'CMA-ES':
        _cma(fc, lb, ub, max_fe, seed)
    else:
        _mealpy(name, fc, lb, ub, max_fe, seed)
    if ret_x is None or not np.isfinite(ret_v):
        ret_v, ret_x = fc.best, fc.best_x
    return {'best': fc.best, 'best_x': fc.best_x, 'returned': float(ret_v), 'returned_x': ret_x, 'nfe': fc.n,
            'trace': fc.trace}


def run(name, f, lb, ub, max_fe, seed):
    """Returns (best value within budget, value returned by the algorithm, evaluations used, convergence trace)."""
    r = run_full(name, f, lb, ub, max_fe, seed)
    return r['best'], r['returned'], r['nfe'], r['trace']
