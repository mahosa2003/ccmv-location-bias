"""Analysis of the ESDO benchmark: tables (JSON) and figures (PNG) for the report.

Uses only complete runs (a run counts when every algorithm has finished it on every problem).
Usage: python analyze.py   ->  results/summary.json, figures/*.png
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
from esdo.algos import ALGOS  # noqa: E402
from esdo.cec17 import FUNCS, NAMES, GROUP  # noqa: E402
from esdo.portfolio import load_uef, load_orlib, CCMV, percentage_errors, MARKETS  # noqa: E402

RES, FIG = os.path.join(HERE, 'results'), os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
# chart tokens (reference palette, light mode)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
EMPH = {'ESDO-R': BLUE, 'ESDO': ORANGE}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9,
                     'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'grid.linestyle': '-', 'axes.axisbelow': True,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'legend.frameon': False,
                     'axes.spines.top': False, 'axes.spines.right': False})


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load_jsonl(path, keys):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=keys, keep='first').reset_index(drop=True)


def complete_runs(df, keys, per_run):
    cnt = df.groupby('run').size()
    if os.environ.get('ESDO_PARTIAL'):          # debugging on unfinished results
        return sorted(cnt.index.tolist())
    return sorted(cnt[cnt == per_run].index.tolist())


# ======================================================================= CEC 2017
def analyze_cec(summary):
    path = os.path.join(RES, 'cec17_D10.jsonl')
    if not os.path.exists(path):
        return
    df = load_jsonl(path, ['fn', 'alg', 'run'])
    runs = complete_runs(df, ['fn', 'alg'], len(FUNCS) * len(ALGOS))
    df = df[df.run.isin(runs)].copy()
    df['err'] = df.err.clip(lower=0.0)
    df.loc[df.err < 1e-8, 'err'] = 0.0
    mean = df.pivot_table(index='fn', columns='alg', values='err', aggfunc='mean')[ALGOS]
    std = df.groupby(['fn', 'alg']).err.std().unstack()[ALGOS].fillna(0.0)
    ranks = mean.rank(axis=1, method='average')
    avg = ranks.mean().sort_values()
    fr = stats.friedmanchisquare(*[mean[a].values for a in ALGOS])
    by_group = {}
    for g in ['Unimodal', 'Simple multimodal', 'Hybrid', 'Composition']:
        idx = [f for f in FUNCS if GROUP[f] == g]
        by_group[g] = ranks.loc[idx].mean().round(2).to_dict()
    # Wilcoxon signed-rank tests over functions, ESDO-R against every other algorithm (Holm-adjusted)
    wil = {}
    others = [a for a in ALGOS if a != 'ESDO-R']
    ps, stats_ = [], []
    for a in others:
        d = mean['ESDO-R'] - mean[a]
        better = int((d < 0).sum()); worse = int((d > 0).sum())
        try:
            p = stats.wilcoxon(mean['ESDO-R'], mean[a], zero_method='zsplit').pvalue
        except ValueError:
            p = 1.0
        ps.append(p); stats_.append((better, worse))
    for a, p, padj, (b, w) in zip(others, ps, holm(ps), stats_):
        wil[a] = {'esdo_r_better': b, 'esdo_r_worse': w, 'p': float(p), 'p_holm': float(padj)}
    # per-function rank-sum tests (only meaningful with >= 5 runs)
    wtl = {}
    if len(runs) >= 5:
        for a in others:
            w = t = l = 0
            for f in FUNCS:
                x = df[(df.fn == f) & (df.alg == 'ESDO-R')].err.values
                y = df[(df.fn == f) & (df.alg == a)].err.values
                if np.allclose(x, y) or (np.ptp(np.r_[x, y]) == 0):
                    t += 1; continue
                p = stats.mannwhitneyu(x, y, alternative='two-sided').pvalue
                if p >= 0.05:
                    t += 1
                elif np.median(x) < np.median(y):
                    w += 1
                else:
                    l += 1
            wtl[a] = [w, t, l]
    summary['cec'] = {'runs': len(runs), 'mean': mean.to_dict(), 'std': std.to_dict(),
                      'avg_rank': avg.round(2).to_dict(), 'friedman_stat': float(fr.statistic),
                      'friedman_p': float(fr.pvalue), 'rank_by_group': by_group, 'wilcoxon_vs_esdo_r': wil,
                      'wtl_esdo_r': wtl, 'names': {int(k): v for k, v in NAMES.items()}}
    # ---- figure: convergence curves on four representative functions
    fns = [1, 5, 13, 22]
    fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.9))
    axs = axs.ravel()
    for ax, f in zip(axs, fns):
        sub = df[df.fn == f]
        for a in ALGOS:
            tr = [np.array(t) for t in sub[sub.alg == a].trace if len(t)]
            if not tr:
                continue
            L = min(len(t) for t in tr)
            fe = tr[0][:L, 0]
            med = np.median(np.vstack([t[:L, 1] for t in tr]), axis=0)
            med = np.maximum(med, 1e-8)
            col = EMPH.get(a, AQUA if a == 'CMA-ES' else MUTED)
            lw = 1.8 if a in EMPH or a == 'CMA-ES' else 0.9
            z = 3 if a in EMPH or a == 'CMA-ES' else 1
            ax.plot(fe / 1000, med, color=col, lw=lw, zorder=z,
                    label=a if (a in EMPH or a == 'CMA-ES') else None)
        ax.set_yscale('log')
        ax.set_title('F%d %s' % (f, NAMES[f]), loc='left')
        ax.set_xlabel('evaluations (thousands)')
    for ax in (axs[0], axs[2]):
        ax.set_ylabel('median error (best so far)')
    axs[0].plot([], [], color=MUTED, lw=0.9, label='other six algorithms')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, 'fig_cec_convergence.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    return avg


# ======================================================================= centre bias
def analyze_center(summary):
    path = os.path.join(RES, 'center.jsonl')
    if not os.path.exists(path):
        return
    df = load_jsonl(path, ['func', 'case', 'alg', 'run'])
    runs = complete_runs(df, ['func', 'case', 'alg'], 8 * len(ALGOS))
    df = df[df.run.isin(runs)].copy()
    df['err'] = df.err.clip(lower=0.0)
    m = df.pivot_table(index=['func', 'alg'], columns='case', values='err', aggfunc='mean')
    out = {}
    for (fn, a), row in m.iterrows():
        out.setdefault(fn, {})[a] = {'centre': float(row['centre']), 'shifted': float(row['shifted'])}
    summary['center'] = {'runs': len(runs), 'D': int(df.D.iloc[0]), 'FE': int(df.FE.iloc[0]), 'mean': out}
    funcs = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank']
    fig, axs = plt.subplots(2, 2, figsize=(6.9, 5.6), sharey=False)
    axs = axs.ravel()
    floor = 1e-10
    x = np.arange(len(ALGOS))
    for ax, fn in zip(axs, funcs):
        c = np.array([max(out[fn][a]['centre'], floor) for a in ALGOS])
        s = np.array([max(out[fn][a]['shifted'], floor) for a in ALGOS])
        for i in range(len(ALGOS)):
            ax.plot([i, i], [c[i], s[i]], color=AXIS, lw=1.0, zorder=1)
        ax.scatter(x, c, s=26, color=BLUE, zorder=3, label='optimum at the centre (x* = 0)',
                   edgecolors='white', linewidths=1.0)
        ax.scatter(x, s, s=26, color=ORANGE, zorder=3, label='optimum shifted (random o)', marker='D',
                   edgecolors='white', linewidths=1.0)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(ALGOS, rotation=60, ha='right', fontsize=7.5)
        ax.set_title(fn + ' (D = %d)' % summary['center']['D'], loc='left')
        ax.grid(axis='x', visible=False)
    for ax in (axs[0], axs[2]):
        ax.set_ylabel('mean error (log scale)')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, 'fig_center_bias.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


# ======================================================================= portfolio
def analyze_port(summary):
    path = os.path.join(RES, 'port.jsonl')
    if not os.path.exists(path):
        return
    df = load_jsonl(path, ['port', 'alg', 'run', 'e'])
    ports = sorted(df.port.unique())
    E = int(df.e.max()) + 1
    per_run = len(ports) * len(ALGOS) * E
    runs = complete_runs(df, ['port', 'alg', 'e'], per_run)
    df = df[df.run.isin(runs)].copy()
    ex = load_jsonl(os.path.join(RES, 'port_exact.jsonl'), ['port', 'e']) if os.path.exists(
        os.path.join(RES, 'port_exact.jsonl')) else pd.DataFrame(columns=['port', 'e', 'obj', 'status'])
    mpath = os.path.join(RES, 'port_mirror.jsonl')
    dm = load_jsonl(mpath, ['port', 'alg', 'run', 'e']) if os.path.exists(mpath) else pd.DataFrame()
    # reference objective per (port, e): best of SCIP incumbent and every run of every algorithm (both encodings)
    ref = {}
    allruns = (pd.concat([df[['port', 'e', 'obj', 'var', 'ret']], dm[['port', 'e', 'obj', 'var', 'ret']]], ignore_index=True)
               if len(dm) else df)
    for (k, e), g in allruns.groupby(['port', 'e']):
        i = g.obj.idxmin()
        best = (float(g.obj.min()), float(g.loc[i, 'var']), float(g.loc[i, 'ret']), 'metaheuristic')
        h = ex[(ex.port == k) & (ex.e == e)]
        if len(h) and float(h.obj.iloc[0]) <= best[0] + 1e-12:
            best = (float(h.obj.iloc[0]), float(h['var'].iloc[0]), float(h.ret.iloc[0]), str(h.status.iloc[0]))
        ref[(k, e)] = best
    lam = df.groupby('e').lam.first().to_dict()
    df['ref'] = [ref[(k, e)][0] for k, e in zip(df.port, df.e)]
    scale = np.array([lam[e] * ref[(k, e)][1] + (1 - lam[e]) * ref[(k, e)][2] for k, e in zip(df.port, df.e)])
    df['relgap'] = (df.obj - df.ref) / scale
    df['hit'] = df.relgap <= 1e-4
    uef = {k: load_uef(k) for k in ports}
    rows = {}
    for (k, a, r), g in df.groupby(['port', 'alg', 'run']):
        pe = percentage_errors(list(zip(g['var'], g.ret)), uef[k])
        rows.setdefault((k, a), []).append((np.nanmean(pe), np.nanmedian(pe), g.relgap.mean(), g.hit.mean()))
    table = {}
    for (k, a), v in rows.items():
        v = np.array(v)
        table.setdefault(int(k), {})[a] = {'mpe': float(v[:, 0].mean()), 'mpe_sd': float(v[:, 0].std(ddof=0)),
                                           'medpe': float(v[:, 1].mean()), 'relgap': float(v[:, 2].mean()),
                                           'hit': float(v[:, 3].mean())}
    # exact / best-known frontier quality
    refpe = {}
    for k in ports:
        pts = [(ref[(k, e)][1], ref[(k, e)][2]) for e in range(E)]
        pe = percentage_errors(pts, uef[k])
        n_opt = int(sum(1 for e in range(E) if ref[(k, e)][3] == 'optimal'))
        refpe[int(k)] = {'mpe': float(np.nanmean(pe)), 'medpe': float(np.nanmedian(pe)), 'proven_optimal': n_opt}
    # Friedman over (port, lambda) blocks using mean objective over runs
    blk = df.pivot_table(index=['port', 'e'], columns='alg', values='obj', aggfunc='mean')[ALGOS]
    rk = blk.rank(axis=1, method='average')
    avg = rk.mean().sort_values()
    fr = stats.friedmanchisquare(*[blk[a].values for a in ALGOS])
    others = [a for a in ALGOS if a != 'ESDO-R']
    ps, cnt = [], []
    for a in others:
        try:
            p = stats.wilcoxon(blk['ESDO-R'], blk[a], zero_method='zsplit').pvalue
        except ValueError:
            p = 1.0
        ps.append(p); cnt.append((int((blk['ESDO-R'] < blk[a] - 1e-12).sum()), int((blk['ESDO-R'] > blk[a] + 1e-12).sum())))
    wil = {a: {'p': float(p), 'p_holm': float(q), 'esdo_r_better': b, 'esdo_r_worse': w}
           for a, p, q, (b, w) in zip(others, ps, holm(ps), cnt)}
    # control experiment: mirrored encoding x -> 1 - x (same problems, apex of the optimal cones moved to (1,...,1))
    mirror = {}
    if len(dm):
        dm = dm[dm.port.isin(ports)].copy()
        sc = np.array([lam[e] * ref[(k, e)][1] + (1 - lam[e]) * ref[(k, e)][2] for k, e in zip(dm.port, dm.e)])
        dm['relgap'] = (dm.obj - np.array([ref[(k, e)][0] for k, e in zip(dm.port, dm.e)])) / sc
        full = dm.groupby(['port', 'alg']).size()
        for (k, a), n in full.items():
            if n < E:
                continue
            std = df[(df.port == k) & (df.alg == a) & (df.run == 0)].relgap
            mirror.setdefault(int(k), {})[a] = {'standard': float(std.mean()), 'mirrored': float(dm[(dm.port == k) & (dm.alg == a)].relgap.mean())}
    summary['port'] = {'runs': len(runs), 'E': E, 'ports': [int(k) for k in ports], 'table': table, 'mirror': mirror,
                       'reference': refpe, 'avg_rank': avg.round(2).to_dict(), 'friedman_stat': float(fr.statistic),
                       'friedman_p': float(fr.pvalue), 'wilcoxon_vs_esdo_r': wil,
                       'n': {int(k): int(load_orlib(k)[0].size) for k in ports},
                       'markets': {int(k): MARKETS[k] for k in ports},
                       'budget': {int(k): int(df[df.port == k].fe.iloc[0]) for k in ports}}
    # ---- figure: frontier for port 1 and port 2 (run 0)
    sel = [k for k in [1, 2] if k in ports]
    fig, axs = plt.subplots(1, len(sel), figsize=(3.45 * len(sel), 3.1))
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, sel):
        u = uef[k]
        ax.plot(u[:, 1] * 1e3, u[:, 0] * 100, color=MUTED, lw=1.4, label='unconstrained frontier', zorder=1)
        pts = np.array([(ref[(k, e)][1], ref[(k, e)][2]) for e in range(E)])
        ax.scatter(pts[:, 0] * 1e3, pts[:, 1] * 100, s=46, facecolors='none', edgecolors=INK, linewidths=1.0,
                   label='best known (SCIP) K = 10', zorder=2)
        r0 = df[(df.port == k) & (df.run == runs[0])]
        for a, col, mk in [('CMA-ES', AQUA, 'o'), ('ESDO-R', BLUE, 's'), ('ESDO', ORANGE, 'D')]:
            g = r0[r0.alg == a]
            ax.scatter(g['var'] * 1e3, g.ret * 100, s=22, color=col, marker=mk, label=a, zorder=3,
                       edgecolors='white', linewidths=0.8)
        ax.set_xlabel('variance of weekly return (×10⁻³)')
        ax.set_ylabel('mean weekly return (%)')
        ax.set_title('port%d: %s, n = %d' % (k, MARKETS[k], load_orlib(k)[0].size), loc='left')
    axs[0].legend(loc='lower right', fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_port_frontier.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    return avg


def fig_ranks(summary):
    panels = [(k, t) for k, t in [('cec', 'CEC 2017, D = 10'),
                                  ('port', 'CCMV portfolios')] if k in summary]
    if not panels:
        return
    fig, axs = plt.subplots(1, len(panels), figsize=(3.45 * len(panels), 2.8))
    axs = np.atleast_1d(axs)
    for ax, (k, title) in zip(axs, panels):
        avg = pd.Series(summary[k]['avg_rank']).sort_values(ascending=False)
        cols = [EMPH.get(a, MUTED) for a in avg.index]
        ax.barh(np.arange(len(avg)), avg.values, color=cols, height=0.62, edgecolor='white', linewidth=2)
        ax.set_yticks(np.arange(len(avg)))
        ax.set_yticklabels(avg.index)
        for i, v in enumerate(avg.values):
            ax.text(v + 0.08, i, '%.2f' % v, va='center', fontsize=7.5, color=INK2)
        ax.set_xlim(0, len(avg) + 0.6)
        ax.set_xlabel('average Friedman rank (1 = best)')
        ax.set_title(title, loc='left')
        ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_ranks.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    summary = {}
    analyze_center(summary)
    analyze_cec(summary)
    analyze_port(summary)
    fig_ranks(summary)
    with open(os.path.join(RES, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, default=float)
    for k in summary:
        print(k, 'runs =', summary[k].get('runs'), 'avg_rank =', summary[k].get('avg_rank'))


if __name__ == '__main__':
    main()
