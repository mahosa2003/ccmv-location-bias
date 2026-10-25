"""Analysis for paper 1 (location bias of metaheuristics and its interaction with the CCMV encoding).

Inputs in results_p1/: center.jsonl, origin.jsonl, port_std.jsonl, port_mirror.jsonl, port_corner.jsonl,
port_exact.jsonl, mechanism.jsonl. Outputs: results_p1/summary_p1.json and figures in figures_p1/.
Only runs that are complete for every algorithm and problem are used (set P1_PARTIAL=1 to inspect partial results).
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings('ignore')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from analyze import load_jsonl, holm, BLUE, ORANGE, AQUA, INK, INK2, MUTED, AXIS, GRID  # noqa: E402  (sets rcParams)
from esdo.algos import ALGOS_P1  # noqa: E402
from esdo.portfolio import load_uef, load_orlib, percentage_errors, MARKETS  # noqa: E402

RES, FIG = os.path.join(HERE, 'results_p1'), os.path.join(HERE, 'figures_p1')
os.makedirs(FIG, exist_ok=True)
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType fonts embedded in the vector copies


def save_fig(fig, name):
    """PNG (220 dpi, embedded in the manuscript) and a vector PDF copy for the journal."""
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=220, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
PARTIAL = bool(os.environ.get('P1_PARTIAL'))
FUNCS = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank']
TINY = 1e-8                      # errors below this count as zero


def complete(df, n_per_run):
    cnt = df.groupby('run').size()
    runs = sorted(cnt.index) if PARTIAL else sorted(cnt[cnt >= n_per_run].index)
    return df[df.run.isin(runs)].copy(), len(runs)


def mw(u, v):
    u, v = np.maximum(u, TINY), np.maximum(v, TINY)
    if np.allclose(u, v[0]) and np.allclose(v, v[0]):
        return 1.0
    try:
        return float(stats.mannwhitneyu(u, v, alternative='two-sided').pvalue)
    except ValueError:
        return 1.0


def better(u, v, factor=10.0, alpha=0.01):
    """u clearly better (smaller error) than v: median error at least `factor` times smaller and p < alpha."""
    mu, mv = max(np.median(u), TINY), max(np.median(v), TINY)
    return bool(mv / mu >= factor and mw(u, v) < alpha)


def bias_tests(S):
    out = {}
    p = os.path.join(RES, 'center.jsonl')
    if os.path.exists(p):
        df = load_jsonl(p, ['func', 'case', 'alg', 'run'])
        df = df[df.alg.isin(ALGOS_P1)]
        df, nr = complete(df, len(FUNCS) * 2 * len(ALGOS_P1))
        df['err'] = df.err.clip(lower=0)
        res, cls = {}, {}
        for (fn, a), x in df.groupby(['func', 'alg']):
            c, s = x[x.case == 'centre'].err.values, x[x.case == 'shifted'].err.values
            res.setdefault(fn, {})[a] = {'centre': float(c.mean()), 'shifted': float(s.mean()),
                                         'centre_med': float(np.median(c)), 'shifted_med': float(np.median(s)), 'p': mw(c, s)}
            cls.setdefault(a, 0)
            cls[a] += better(c, s)
        out['center'] = {'runs': nr, 'mean': res, 'n_funcs_centre_better': cls}
    p = os.path.join(RES, 'origin.jsonl')
    if os.path.exists(p):
        df = load_jsonl(p, ['func', 'case', 'alg', 'run'])
        df = df[df.alg.isin(ALGOS_P1)]
        cases = [c for c in ['origin', 'corner', 'offcorner', 'centre', 'random'] if c in set(df.case)]
        df, nr = complete(df, len(FUNCS) * len(cases) * len(ALGOS_P1))
        df['err'] = df.err.clip(lower=0)
        res, cls, flags = {}, {}, {}
        for (fn, a), x in df.groupby(['func', 'alg']):
            e = {c: x[x.case == c].err.values for c in cases}
            fl = flags.setdefault(a, {}).setdefault(fn, {})
            for c1, c2 in [('origin', 'corner'), ('corner', 'origin'), ('corner', 'offcorner'), ('origin', 'offcorner'),
                           ('origin', 'random'), ('corner', 'random'), ('offcorner', 'random'), ('centre', 'random')]:
                if c1 in e and c2 in e:
                    fl[c1 + '_vs_' + c2] = better(e[c1], e[c2])
            rec = {c: float(e[c].mean()) for c in cases}
            rec.update({c + '_med': float(np.median(e[c])) for c in cases})
            res.setdefault(fn, {})[a] = rec
            k = cls.setdefault(a, {'origin_vs_random': 0, 'corner_vs_random': 0, 'offcorner_vs_random': 0, 'origin_vs_corner': 0,
                                   'corner_vs_origin': 0, 'corner_vs_offcorner': 0, 'origin_vs_offcorner': 0, 'centre_vs_random': 0})
            k['origin_vs_random'] += better(e['origin'], e['random'])
            k['centre_vs_random'] += better(e['centre'], e['random'])
            if 'corner' in e:
                k['corner_vs_random'] += better(e['corner'], e['random'])
                k['origin_vs_corner'] += better(e['origin'], e['corner'])
                k['corner_vs_origin'] += better(e['corner'], e['origin'])
            if 'offcorner' in e:
                k['offcorner_vs_random'] += better(e['offcorner'], e['random'])
                k['corner_vs_offcorner'] += better(e['corner'], e['offcorner'])
                k['origin_vs_offcorner'] += better(e['origin'], e['offcorner'])
        out['origin'] = {'runs': nr, 'cases': cases, 'mean': res, 'counts': cls, 'flags': flags}
    S.update(out)


def reference(frames, ex, strong=None):
    """Best known solution per (port, e): the best of all runs and of the solver (plain 60 s model and, where
    available, the strengthened long run). Solver solutions are scored by evaluating the objective at their weights
    (lambda * var - (1 - lambda) * ret, from the stored weights): the value SCIP reports uses the epigraph variable t,
    which may lie below w'Sigma w within the feasibility tolerance of the quadratic constraint.
    Returns {(port, e): (f, var, ret, status, source)} with status 'optimal' when a solver proved optimality and
    source 'solver' or 'metaheuristic' for whichever portfolio is the best known."""
    allr = pd.concat([f[['port', 'e', 'lam', 'obj', 'var', 'ret']] for f in frames.values()], ignore_index=True)
    ref = {}
    for (k, e), g in allr.groupby(['port', 'e']):
        i = g.obj.idxmin()
        best = [float(g.obj.min()), float(g.loc[i, 'var']), float(g.loc[i, 'ret']), 'open', 'metaheuristic']
        lam = float(g.lam.iloc[0])
        proven = False
        for src in [ex, strong]:
            if src is None or not len(src):
                continue
            h = src[(src.port == k) & (src.e == e)]
            if not len(h):
                continue
            proven = proven or str(h.status.iloc[0]) == 'optimal'
            if pd.isna(h['var'].iloc[0]):
                continue
            v, r = float(h['var'].iloc[0]), float(h.ret.iloc[0])
            ftrue = lam * v - (1 - lam) * r
            if ftrue < best[0] - 1e-12:
                best = [ftrue, v, r, best[3], 'solver']
        best[3] = 'optimal' if proven else 'open'
        ref[(k, e)] = tuple(best)
    return ref


def gaps(S):
    ex = load_jsonl(os.path.join(RES, 'port_exact.jsonl'), ['port', 'e'])
    frames = {}
    for enc in ['std', 'mirror', 'corner']:
        p = os.path.join(RES, 'port_%s.jsonl' % enc)
        if os.path.exists(p):
            d = load_jsonl(p, ['port', 'alg', 'run', 'e'])
            frames[enc] = d[d.alg.isin(ALGOS_P1)].copy()
    extra = {}
    for enc in ['std_b1000', 'corner_b1000']:
        p = os.path.join(RES, 'port_%s.jsonl' % enc)
        if os.path.exists(p):
            d = load_jsonl(p, ['port', 'alg', 'run', 'e'])
            extra[enc] = d[d.alg.isin(ALGOS_P1)].copy()
    sp = os.path.join(RES, 'port_exact_strong.jsonl')
    strong = load_jsonl(sp, ['port', 'e']) if os.path.exists(sp) else None
    ref = reference({**frames, **extra}, ex, strong)       # the best known solution includes the larger-budget runs
    res = {}
    for enc, d in list(frames.items()):
        ports = sorted(d.port.unique())
        d, nr = complete(d, len(ports) * 20 * len(ALGOS_P1))
        if nr == 0:
            frames.pop(enc)
            continue
        sc = np.array([lam * ref[(k, e)][1] + (1 - lam) * ref[(k, e)][2] for lam, k, e in zip(d.lam, d.port, d.e)])
        d['gap'] = (d.obj - np.array([ref[(k, e)][0] for k, e in zip(d.port, d.e)])) / sc
        d['hit'] = d.gap <= 1e-4
        frames[enc] = d
        tab = d.groupby(['port', 'alg']).gap.mean().unstack()
        hit = d.groupby(['port', 'alg']).hit.mean().unstack()
        blk = d.pivot_table(index=['port', 'e'], columns='alg', values='obj', aggfunc='mean')[ALGOS_P1]
        rk = blk.rank(axis=1).mean().sort_values()
        res[enc] = {'runs': nr, 'ports': [int(k) for k in ports],
                    'gap': {int(k): tab.loc[k].to_dict() for k in tab.index},
                    'hit': {int(k): hit.loc[k].to_dict() for k in hit.index},
                    'gap_all': d.groupby('alg').gap.mean().to_dict(), 'gap_median_all': d.groupby('alg').gap.median().to_dict(),
                    'avg_rank': rk.round(2).to_dict()}
    if 'std' in frames:
        d = frames['std']
        ports = res['std']['ports']
        uef = {k: load_uef(k) for k in ports}
        mpe = {}
        for (k, a, r), g in d.groupby(['port', 'alg', 'run']):
            pe = percentage_errors(list(zip(g['var'], g.ret)), uef[k])
            mpe.setdefault((k, a), []).append(np.nanmean(pe))
        res['std']['mpe'] = {int(k): {a: float(np.mean(v)) for (kk, a), v in mpe.items() if kk == k} for k in ports}
        refm = {}
        for k in ports:
            pts = [(ref[(k, e)][1], ref[(k, e)][2]) for e in range(20)]
            refm[int(k)] = {'mpe': float(np.nanmean(percentage_errors(pts, uef[k]))),
                            'proven_optimal': int(sum(ref[(k, e)][3] == 'optimal' for e in range(20))),
                            'scip_best': int(sum(ref[(k, e)][4] == 'solver' for e in range(20)))}
        res['reference'] = refm
        res['open_refs'] = [{'port': int(k), 'e': int(e), 'lam': float(e / 19), 'source': v[4]}
                            for (k, e), v in sorted(ref.items()) if v[3] != 'optimal']
        # the solver's own best portfolio, scored at its weights, against the best known value (units of Eq. (5))
        sol = {}
        for src in [ex, strong]:
            if src is None or not len(src):
                continue
            for r in src.itertuples():
                if pd.isna(r.var):
                    continue
                lam_r = float(r.e) / 19
                fv = lam_r * float(r.var) - (1 - lam_r) * float(r.ret)
                key = (int(r.port), int(r.e))
                sol[key] = min(sol.get(key, np.inf), fv)
        gsol = {key: (sol[key] - ref[key][0]) / ((key[1] / 19) * ref[key][1] + (1 - key[1] / 19) * ref[key][2]) for key in sol}
        proven_keys = [key for key in gsol if ref[key][3] == 'optimal']
        res['solver_vs_ref'] = {'n_within_hit': int(sum(v <= 1e-4 for v in gsol.values())), 'n_pairs': len(gsol),
                                'n_improved_on_proven': int(sum(gsol[key] > 1e-9 for key in proven_keys)),
                                'max_improvement_on_proven': float(max([gsol[key] for key in proven_keys] + [0.0]))}
        if strong is not None and len(strong):
            # gap between the best known value and the best lower bound (plain or strengthened model), in the units of
            # Eq. (5): the objective values themselves can be close to zero near lambda = 0.9
            b60 = {(int(r.port), int(r.e)): float(r.bound) for r in ex.itertuples()}
            res['strong'] = []
            for r in strong.itertuples():
                k, e, lam = int(r.port), int(r.e), float(r.lam)
                lb = max(float(r.bound), b60.get((k, e), -np.inf))
                fv, vv, rv = ref[(k, e)][:3]
                res['strong'].append({'port': k, 'e': e, 'lam': lam, 'status': str(r.status),
                                      'obj': (None if pd.isna(r.obj) else float(r.obj)), 'bound': float(r.bound), 'best_bound': lb,
                                      'ref': fv, 'time': float(r.time),
                                      'rel_gap': float(max(fv - lb, 0.0) / (lam * vv + (1 - lam) * rv))})
        # rank agreement between MPE and the gap to the best known solution
        mrank = pd.DataFrame(res['std']['mpe']).T[ALGOS_P1].rank(axis=1).mean()
        grank = pd.DataFrame(res['std']['gap']).T[ALGOS_P1].rank(axis=1).mean()
        rho = stats.spearmanr(mrank, grank)
        res['mpe_vs_gap'] = {'spearman': float(rho.statistic), 'p': float(rho.pvalue),
                             'mpe_rank': mrank.round(2).to_dict(), 'gap_rank': grank.round(2).to_dict()}
        # exact-solution MPE versus algorithm MPE (is the exact frontier's MPE a floor?)
        below = {k: [a for a in ALGOS_P1 if res['std']['mpe'][k][a] < refm[k]['mpe'] - 1e-9] for k in res['std']['mpe']}
        res['mpe_below_reference'] = below
    for enc in ['mirror', 'corner']:
        if 'std' in frames and enc in frames:
            a, b = frames['std'], frames[enc]
            common = sorted(set(a.port) & set(b.port))
            m = a[a.port.isin(common)].merge(b[b.port.isin(common)], on=['port', 'alg', 'run', 'e'], suffixes=('_s', '_m'))
            pairs, ps = {}, []
            for alg in ALGOS_P1:
                g = m[m.alg == alg]
                try:
                    pv = float(stats.wilcoxon(g.gap_s, g.gap_m, zero_method='zsplit').pvalue)
                except ValueError:
                    pv = 1.0
                ps.append(pv)
                pairs[alg] = {'n': int(len(g)), 'gap_std': float(g.gap_s.mean()), 'gap_other': float(g.gap_m.mean()),
                              'ratio': float(g.gap_m.mean() / max(g.gap_s.mean(), 1e-12)), 'p': pv,
                              'worse_share': float((g.gap_m > g.gap_s + 1e-12).mean()),
                              'per_port': {int(k): [float(g[g.port == k].gap_s.mean()), float(g[g.port == k].gap_m.mean())] for k in common}}
            for alg, q in zip(ALGOS_P1, holm(ps)):
                pairs[alg]['p_holm'] = float(q)
            res['std_vs_' + enc] = {'ports': [int(k) for k in common], 'pairs': pairs}
    # budget check: the standard and randomly reflected encodings with 1000 n evaluations
    if 'std_b1000' in extra and 'corner_b1000' in extra:
        bb = {}
        for enc, d in extra.items():
            ports_b = sorted(d.port.unique())
            d, nr = complete(d, len(ports_b) * 20 * len(ALGOS_P1))
            if nr == 0:
                continue
            sc = np.array([lam * ref[(k, e)][1] + (1 - lam) * ref[(k, e)][2] for lam, k, e in zip(d.lam, d.port, d.e)])
            d['gap'] = (d.obj - np.array([ref[(k, e)][0] for k, e in zip(d.port, d.e)])) / sc
            extra[enc] = d
            bb[enc] = {'runs': nr, 'ports': [int(k) for k in ports_b], 'fe': {int(k): int(d[d.port == k].fe.iloc[0]) for k in ports_b},
                       'gap': {int(k): d[d.port == k].groupby('alg').gap.mean().to_dict() for k in ports_b}}
        if 'std_b1000' in bb and 'corner_b1000' in bb:
            a, b = extra['std_b1000'], extra['corner_b1000']
            m = a.merge(b, on=['port', 'alg', 'run', 'e'], suffixes=('_s', '_m'))
            pairs, ps = {}, []
            for alg in ALGOS_P1:
                g = m[m.alg == alg]
                try:
                    pv = float(stats.wilcoxon(g.gap_s, g.gap_m, zero_method='zsplit').pvalue)
                except ValueError:
                    pv = 1.0
                ps.append(pv)
                pairs[alg] = {'n': int(len(g)), 'gap_std': float(g.gap_s.mean()), 'gap_other': float(g.gap_m.mean()),
                              'ratio': float(g.gap_m.mean() / max(g.gap_s.mean(), 1e-12)), 'p': pv,
                              'per_port': {int(k): [float(g[g.port == k].gap_s.mean()), float(g[g.port == k].gap_m.mean())]
                                           for k in sorted(g.port.unique())}}
            for alg, q in zip(ALGOS_P1, holm(ps)):
                pairs[alg]['p_holm'] = float(q)
            bb['std_vs_corner'] = pairs
            # the same comparison with the default budget, restricted to the same instances and runs
            if 'std' in frames and 'corner' in frames:
                a0 = frames['std'][frames['std'].port.isin(bb['std_b1000']['ports'])]
                b0 = frames['corner'][frames['corner'].port.isin(bb['std_b1000']['ports'])]
                m0 = a0.merge(b0, on=['port', 'alg', 'run', 'e'], suffixes=('_s', '_m'))
                bb['default_same_ports'] = {alg: {'gap_std': float(m0[m0.alg == alg].gap_s.mean()),
                                                  'gap_other': float(m0[m0.alg == alg].gap_m.mean()),
                                                  'ratio': float(m0[m0.alg == alg].gap_m.mean() / max(m0[m0.alg == alg].gap_s.mean(), 1e-12))}
                                            for alg in ALGOS_P1}
        res['budget1000'] = bb
    S['portfolio'] = res
    S['n_assets'] = {int(k): int(load_orlib(k)[0].size) for k in range(1, 6)}
    S['markets'] = {int(k): MARKETS[k] for k in range(1, 6)}
    return frames, ref


def mechanism(S):
    p = os.path.join(RES, 'mechanism.jsonl')
    if not os.path.exists(p):
        return
    rows = [json.loads(l) for l in open(p) if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates(['port', 'lam', 'alg', 'enc', 'run'])
    out = {}
    for (k, a, enc), g in df.groupby(['port', 'alg', 'enc']):
        tr = np.array([[t[1] for t in r] for r in g.trace if len(r)])
        fe = [t[0] for t in g.trace.iloc[0]]
        out.setdefault(int(k), {}).setdefault(a, {})[enc] = {
            'runs': int(len(g)), 'fe': fe, 'mean_x': tr.mean(axis=0).round(4).tolist(), 'final_mean_x': float(g.final_mean_x.mean()),
            'obj': float(g.obj.mean())}
    S['mechanism'] = out


def figures(S):
    # Figure 1: centre-bias test
    if 'center' in S:
        Z = S['center']['mean']
        fig, axs = plt.subplots(2, 2, figsize=(6.9, 5.6))
        x = np.arange(len(ALGOS_P1))
        for ax, fn in zip(axs.ravel(), FUNCS):
            c = np.array([max(Z[fn][a]['centre_med'], 1e-10) for a in ALGOS_P1])
            s = np.array([max(Z[fn][a]['shifted_med'], 1e-10) for a in ALGOS_P1])
            for i in range(len(x)):
                ax.plot([i, i], [c[i], s[i]], color=AXIS, lw=1.0, zorder=1)
            ax.scatter(x, c, s=24, color=BLUE, zorder=3, label='optimum at the centre of the box', edgecolors='white', linewidths=1)
            ax.scatter(x, s, s=24, color=ORANGE, marker='D', zorder=3, label='optimum shifted at random', edgecolors='white', linewidths=1)
            ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(ALGOS_P1, rotation=60, ha='right', fontsize=7)
            ax.set_title(fn, loc='left'); ax.grid(axis='x', visible=False)
        axs[0, 0].set_ylabel('median error (log scale)'); axs[1, 0].set_ylabel('median error (log scale)')
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_fig(fig, 'fig1_center_bias'); plt.close(fig)
    # Figure 2: origin, opposite corner, centre and random optimum in [0, 200]^D
    if 'origin' in S:
        O = S['origin']['mean']
        cases = S['origin']['cases']
        sty = {'origin': (BLUE, 'o', 'origin'), 'corner': (AQUA, 's', 'opposite corner'),
               'offcorner': ('#8e44ad', 'v', 'off-diagonal corner'), 'centre': (MUTED, '^', 'centre'),
               'random': (ORANGE, 'D', 'random point')}
        fig, axs = plt.subplots(2, 2, figsize=(6.9, 5.6))
        x = np.arange(len(ALGOS_P1))
        off = {c: (j - (len(cases) - 1) / 2) * 0.17 for j, c in enumerate(cases)}
        for ax, fn in zip(axs.ravel(), FUNCS):
            for c in cases:
                v = np.array([max(O[fn][a][c + '_med'], 1e-10) for a in ALGOS_P1])
                col, mk, lab = sty[c]
                ax.scatter(x + off[c], v, s=18, color=col, marker=mk, label=lab, zorder=3, edgecolors='white', linewidths=0.7)
            ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(ALGOS_P1, rotation=60, ha='right', fontsize=7)
            ax.set_title(fn, loc='left'); ax.grid(axis='x', visible=False)
        axs[0, 0].set_ylabel('median error (log scale)'); axs[1, 0].set_ylabel('median error (log scale)')
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7.5)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_fig(fig, 'fig2_origin_corner'); plt.close(fig)
    P = S.get('portfolio', {})
    # Figure 3: CCMV gap, direct versus reflected encodings (two panels)
    if 'std_vs_mirror' in P:
        panels = [(k, t) for k, t in [('std_vs_mirror', 'mirrored encoding (x → 1 − x)'), ('std_vs_corner', 'randomly reflected encoding')] if k in P]
        ratio_max = {a: max(P[k]['pairs'][a]['ratio'] for k, _ in panels) for a in ALGOS_P1}
        algs = sorted(ALGOS_P1, key=lambda a: -ratio_max[a])
        fig, axs = plt.subplots(1, len(panels), figsize=(6.9, 3.1), sharey=True, squeeze=False)
        for ax, (k, t) in zip(axs[0], panels):
            sv = P[k]['pairs']
            for i, a in enumerate(algs):
                s0, m0 = 100 * max(sv[a]['gap_std'], 1e-6), 100 * max(sv[a]['gap_other'], 1e-6)
                ax.plot([i, i], [s0, m0], color=AXIS, lw=1.2, zorder=1)
                ax.scatter([i], [s0], s=26, color=BLUE, zorder=3, edgecolors='white', linewidths=0.8)
                ax.scatter([i], [m0], s=26, color=ORANGE, marker='D', zorder=3, edgecolors='white', linewidths=0.8)
            ax.set_yscale('log'); ax.set_xticks(range(len(algs))); ax.set_xticklabels(algs, rotation=60, ha='right', fontsize=7)
            ax.set_title(t, loc='left'); ax.grid(axis='x', visible=False)
            ax.set_ylim(5, 80)
            ax.set_yticks([5, 10, 20, 40, 80]); ax.set_yticklabels(['5', '10', '20', '40', '80']); ax.minorticks_off()
        axs[0, 0].set_ylabel('mean gap to the reference (%)')
        axs[0, 0].scatter([], [], s=26, color=BLUE, label='direct encoding')
        axs[0, 0].scatter([], [], s=26, color=ORANGE, marker='D', label='reflected encoding')
        axs[0, 0].legend(loc='upper right', fontsize=7.5, frameon=False)
        fig.tight_layout(); save_fig(fig, 'fig3_encoding'); plt.close(fig)
    # Figure 4: where the optimizers search (mean coordinate of evaluated points), one row per instance
    M = S.get('mechanism', {})
    if M:
        ks = sorted(M, key=int)
        sel = [a for a in ['GWO', 'HHO', 'WOA', 'SCA', 'ESDO', 'DE', 'CMA-ES', 'ESDO-R'] if a in M[ks[0]]]
        cols = {'GWO': ORANGE, 'HHO': '#c0392b', 'WOA': '#d4a017', 'SCA': '#8e44ad', 'ESDO': '#7f5539',
                'DE': BLUE, 'CMA-ES': AQUA, 'ESDO-R': '#34495e'}
        fig, axs = plt.subplots(len(ks), 2, figsize=(6.9, 2.5 * len(ks)), sharey=True, squeeze=False)
        for r, k in enumerate(ks):
            for a in sel:
                fe = np.array(M[k][a]['std']['fe'])
                ls = '-' if a in ('GWO', 'HHO', 'WOA', 'SCA', 'ESDO') else '--'
                axs[r, 0].plot(fe, M[k][a]['std']['mean_x'], color=cols[a], lw=1.3, ls=ls, label=a)
                axs[r, 1].plot(fe, 1 - np.array(M[k][a]['mirror']['mean_x']), color=cols[a], lw=1.3, ls=ls, label=a)
            axs[r, 0].set_title('port%s, direct encoding: mean of x' % k, loc='left')
            axs[r, 1].set_title('port%s, mirrored encoding: mean of 1 − x' % k, loc='left')
            axs[r, 0].set_ylabel('distance from apex')
            for ax in axs[r]:
                ax.grid(axis='x', visible=False)
        for ax in axs[-1]:
            ax.set_xlabel('function evaluations')
        axs[0, 1].legend(loc='upper right', fontsize=6.8, frameon=False, ncol=2)
        fig.tight_layout(); save_fig(fig, 'fig4_mechanism'); plt.close(fig)


def main():
    S = {'algorithms': ALGOS_P1}
    bias_tests(S)
    gaps(S)
    mechanism(S)
    figures(S)
    with open(os.path.join(RES, 'summary_p1.json'), 'w') as fh:
        json.dump(S, fh, indent=1, default=float)
    print({k: (v.get('runs') if isinstance(v, dict) else None) for k, v in S.items() if isinstance(v, dict)})
    P = S.get('portfolio', {})
    for enc in ['mirror', 'corner']:
        if 'std_vs_' + enc in P:
            print('== standard vs', enc, P['std_vs_' + enc]['ports'])
            for a, r in sorted(P['std_vs_' + enc]['pairs'].items(), key=lambda x: -x[1]['ratio']):
                print('%-8s gap std %.3f%% %s %.3f%% ratio %.1f p_holm=%.2g' % (a, 100 * r['gap_std'], enc, 100 * r['gap_other'], r['ratio'], r['p_holm']))
    if 'origin' in S:
        print('== origin test counts (functions out of 4):')
        for a, c in S['origin']['counts'].items():
            print('%-8s %s' % (a, c))
    if 'center' in S:
        print('== centre better than shifted (functions out of 4):', S['center']['n_funcs_centre_better'])


if __name__ == '__main__':
    main()
