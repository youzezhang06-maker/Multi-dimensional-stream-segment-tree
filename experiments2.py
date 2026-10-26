"""
Two tasks on the same query design (--task):
  complexity  query-time complexity of the multidimensional TucketTree (test_tucket1.py), checked by fitting,
              with the baselines on the same queries (default)
  error_k     reconstruction error vs the number k of query modes, tree vs Tucker-ALS on the same boxes
              (see run_error_k)

Design (as agreed with the advisor):
  * Every queried mode (time and each static mode with a tree) gets a fixed set of length levels,
    chosen per data set with rank < L < mode size. Static levels: up to --n_static_levels integer
    lengths spread geometrically over [r+1, M-1] (all of them if there are fewer). Time levels:
    --time_min * --time_ratio^j, at most --time_max_frac * T (a window of almost the whole stream is
    covered by a few large nodes, which would bend the curve down at the end).
  * The query table is the FULL combination of the levels of all modes (a p-dimensional table), not
    one mode at a time and not the diagonal: only the interior of the table separates a product
    (log Lt * log L1 * log L2) from a sum (log Lt + log L1 + log L2).
  * Time complexity is not a function of the lengths alone: the same lengths at other endpoints give
    other numbers of canonical pieces. Every cell therefore gets --n_reps random endpoints, drawn
    independently per mode (uniform start in [0, M - L]; if a mode has <= n_reps positions, all of them
    are used in balanced order), and the cell value is the mean over those queries.
  * All queries are generated ONCE per data set, saved to queries_<name>.json, and every method
    (the tree and the baselines) answers exactly the same queries. --queries_dir reuses saved queries.
    Queries are executed in a fixed random order so that any slow drift of the machine does not line up
    with the cells.
  * Per query we record wall time (median of --time_reps runs), hits, partial hits, pieces per static
    mode, leaf pieces (= sum over hits of prod_i s_i) and STITCH / partial calls.
  * The cell means are fitted by several candidate models (constant, sum of logs, product of logs,
    log Lt + log L1 log L2, product of logs + box volume, sum of lengths, box volume). Models are
    compared by leave-one-cell-out RMSE (LOOCV), R^2 and BIC.

Methods (same queries):
  ours     the multidimensional TucketTree, query_dims = [1, 2]
  tucket   the original TUCKET (time tree only); the static ranges are cut afterwards from the factor
           matrices (tucker_partial with QR on each static mode)
  als      Tucker-ALS run from scratch on the queried box, with the tree's settings (--tol, --maxiters)
  The accuracy reference for Eq. 29 (err_method / err_ref - 1) is a tight Tucker-ALS on the same box
  (--als_tol, --als_maxiters); it is not timed.

Inputs: inputs/grid/<name>.npy (already cropped: full time span; static modes chosen by data coverage,
see <name>.json next to it).

Usage (from the TUCKET root):
  python experiments2.py --quick
  python experiments2.py --datasets Traffic
  python experiments2.py --datasets Traffic --queries_dir results/grid-<stamp>     # reuse queries
  python experiments2.py --task error_k --datasets Traffic --out results/errk         # error vs k
"""
import argparse
import csv
import json
import os
import os.path as osp
import time
import zlib
from datetime import datetime

import numpy as np
import torch

from inc.utils import set_seed
from inc.tucker import *
from test_tucket1 import TucketTree

ALL = ['AirQuality', 'Traffic', 'USStock', 'KRStock']
ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
ap.add_argument('--datasets', nargs = '+', default = ALL)
ap.add_argument('--data_root', default = 'inputs/grid')
ap.add_argument('--out_root', default = 'results')
ap.add_argument('--out', default = '', help = 'output folder (default: results/grid-<stamp>)')
ap.add_argument('--queries_dir', default = '', help = 'reuse queries_<name>.json from this folder')
ap.add_argument('--methods', default = 'ours,tucket,als')
ap.add_argument('--T', type = int, default = 0, help = '0 = full time span')
ap.add_argument('--time_rank', type = int, default = 5)
ap.add_argument('--static_rank_div', type = int, default = 4)
ap.add_argument('--time_min', type = int, default = 16)
ap.add_argument('--time_ratio', type = int, default = 2)
ap.add_argument('--time_max_frac', type = float, default = 0.25,
                help = 'time levels stay <= this fraction of T: a window close to T is almost the whole stream and needs fewer hits')
ap.add_argument('--n_static_levels', type = int, default = 6)
ap.add_argument('--n_reps', type = int, default = 30, help = 'random endpoints per cell')
ap.add_argument('--time_reps', type = int, default = 3, help = 'timed runs per query (median)')
ap.add_argument('--tol', type = float, default = 1e-2)
ap.add_argument('--maxiters', type = int, default = 20)
ap.add_argument('--als_tol', type = float, default = 1e-6)
ap.add_argument('--als_maxiters', type = int, default = 200)
ap.add_argument('--prune', type = float, default = 0.7)
ap.add_argument('--threads', type = int, default = 1)
ap.add_argument('--seed', type = int, default = 998244353)
ap.add_argument('--task', default = 'complexity', choices = ['complexity', 'error_k'],
                help = 'complexity: the Lt x L1 x L2 table with the baselines; '
                       'error_k: reconstruction error vs the number k of query modes, tree vs Tucker-ALS')
ap.add_argument('--quick', action = 'store_true', help = 'T=256, 2 endpoints per cell, 1 timed run')
args = ap.parse_args()
if args.quick:
    args.T, args.n_reps, args.time_reps = args.T or 256, 2, 1
METHODS = args.methods.split(',')
torch.set_num_threads(args.threads)
OUT = args.out or osp.join(args.out_root, ('grid-' if args.task == 'complexity' else 'errk-') + datetime.now().strftime('%Y%m%d-%H%M%S') + ('-quick' if args.quick else ''))
os.makedirs(OUT, exist_ok = True)


def log(msg):
    print(msg, flush = True)
    with open(osp.join(OUT, 'log.txt'), 'a') as f:
        f.write(msg + '\n')


# ------------------------------------------------------------------ design
def static_levels(M, r):
    lo, hi = r + 1, M - 1
    if hi - lo + 1 <= args.n_static_levels:
        return list(range(lo, hi + 1))
    return sorted({int(round(x)) for x in np.geomspace(lo, hi, args.n_static_levels)})


def time_levels(T, rt):
    out, L = [], args.time_min
    while L <= args.time_max_frac * T:
        if L > rt:
            out.append(L)
        L *= args.time_ratio
    return out


def starts(rng, n_pos, n):
    """n start positions in [0, n_pos): all positions in balanced random order if n_pos <= n, else distinct"""
    if n_pos <= n:
        s = np.resize(rng.permutation(n_pos), n)
        rng.shuffle(s)
        return s
    return rng.choice(n_pos, size = n, replace = False)


def design(name, sizes, ranks):
    T = sizes[0]
    levels = [time_levels(T, ranks[0])] + [static_levels(sizes[m], ranks[m]) for m in (1, 2)]
    rng = np.random.default_rng([args.seed, zlib.crc32(name.encode())])
    qs = []
    for Lt in levels[0]:
        for L1 in levels[1]:
            for L2 in levels[2]:
                cell = (Lt, L1, L2)
                st = [starts(rng, sizes[m] - cell[m] + 1, args.n_reps) for m in range(3)]  # independent per mode
                for j in range(args.n_reps):
                    t0, a1, a2 = (int(st[m][j]) for m in range(3))
                    qs.append(dict(Lt = Lt, L1 = L1, L2 = L2, rep = j, t0 = t0, t1 = t0 + Lt,
                                   r1 = [a1, a1 + L1], r2 = [a2, a2 + L2]))
    for q in qs:
        assert 0 <= q['t0'] < q['t1'] <= T and ranks[0] < q['Lt'] < T
        for m, key in ((1, 'r1'), (2, 'r2')):
            lo, hi = q[key]
            assert 0 <= lo < hi <= sizes[m] and ranks[m] < hi - lo < sizes[m]
    order = rng.permutation(len(qs)).tolist()      # execution order
    return dict(levels = levels, queries = qs, order = order)


# ------------------------------------------------------------------ helpers
def now():
    return time.perf_counter()


def err64(Xq, tk):
    Xh = tensor_mats_mul(tk.G.double(), A_dim_list = [(U.double(), p) for p, U in enumerate(tk.U)])
    return float(torch.linalg.norm((Xq.double() - Xh).flatten()))


def timed(fn):
    ts, out = [], None
    for _ in range(args.time_reps):
        set_seed(args.seed + 1)
        reset_op_count()
        tic = now()
        out = fn()
        ts.append(now() - tic)
    return out, 1e3 * float(np.median(ts))


def build(X, ranks, query_dims):
    tree = TucketTree(ranks = ranks, tol = args.tol, maxiters = args.maxiters, query_dims = query_dims)
    set_seed(args.seed)
    reset_op_count()
    tic = now()
    for t in range(X.size(0)):
        tree.append(X[t])
        if (t + 1) % 256 == 0:
            log(f'    built {t + 1}/{X.size(0)} slices, {now() - tic:.0f}s')
    return tree, now() - tic, dict(OP_COUNT)


# ------------------------------------------------------------------ fitting
def model_specs():
    """name -> feature function of (Lt, L1, L2) arrays; every model has an intercept"""
    lg = np.log2
    return {
        'const': lambda a, b, c: [],
        'sum log': lambda a, b, c: [lg(a), lg(b), lg(c)],
        'product log (claim)': lambda a, b, c: [lg(a) * lg(b) * lg(c)],
        'log Lt + log L1*log L2': lambda a, b, c: [lg(a), lg(b) * lg(c)],
        'product log + volume': lambda a, b, c: [lg(a) * lg(b) * lg(c), a * b * c],
        'sum linear': lambda a, b, c: [a, b, c],
        'volume': lambda a, b, c: [a * b * c],
    }


def fit_models(cells, y):
    a, b, c = (np.asarray(v, dtype = float) for v in zip(*cells))
    y = np.asarray(y, dtype = float)
    n, out = len(y), {}
    for name, f in model_specs().items():
        A = np.column_stack([np.ones(n)] + f(a, b, c))
        coef, *_ = np.linalg.lstsq(A, y, rcond = None)
        pred = A @ coef
        res = y - pred
        H = A @ np.linalg.pinv(A.T @ A) @ A.T
        loo = res / np.clip(1 - np.diag(H), 1e-12, None)
        rss = float(res @ res)
        k = A.shape[1]
        out[name] = dict(coef = coef.tolist(), r2 = 1 - rss / float(((y - y.mean()) ** 2).sum() or 1),
                         loocv_rmse = float(np.sqrt(np.mean(loo ** 2))),
                         loocv_rel = float(np.sqrt(np.mean(loo ** 2)) / max(abs(y.mean()), 1e-12)),
                         bic = float(n * np.log(max(rss, 1e-300) / n) + k * np.log(n)), k = k, pred = pred.tolist())
    # nonlinear: the product of per-mode affine terms, y = a + c * prod_i (g(L_i) + d_i), g = log2 or identity.
    # 'sum log' above (a + sum_i b_i log2 L_i) is its additive counterpart with per-mode terms of the same kind.
    from scipy.optimize import least_squares
    X3 = np.column_stack([a, b, c])
    for name, g in (('product of (log L_i + d_i) (claim)', np.log2), ('product of (L_i + d_i)', lambda v: v)):
        G = g(X3)

        def fitnl(idx):
            def f(p):
                return p[0] + p[1] * np.prod(G[idx] + p[2:], axis = 1) - y[idx]
            p0 = np.r_[y[idx].min(), (y[idx].max() - y[idx].min()) / max(np.ptp(np.prod(G[idx], axis = 1)), 1e-9), np.zeros(3)]
            return least_squares(f, p0, max_nfev = 5000).x
        p = fitnl(np.arange(n))
        pred = p[0] + p[1] * np.prod(G + p[2:], axis = 1)
        res = y - pred
        loo = np.array([y[i] - (lambda q: q[0] + q[1] * np.prod(G[i] + q[2:]))(fitnl(np.r_[0:i, i + 1:n])) for i in range(n)])
        rss = float(res @ res)
        out[name] = dict(coef = p.tolist(), r2 = 1 - rss / float(((y - y.mean()) ** 2).sum() or 1),
                         loocv_rmse = float(np.sqrt(np.mean(loo ** 2))),
                         loocv_rel = float(np.sqrt(np.mean(loo ** 2)) / max(abs(y.mean()), 1e-12)),
                         bic = float(n * np.log(max(rss, 1e-300) / n) + 5 * np.log(n)), k = 5, pred = pred.tolist())
    best = min(out, key = lambda m: out[m]['loocv_rmse'])
    return out, best


# ------------------------------------------------------------------ main
FIELDS = ['dataset', 'qid', 'Lt', 'L1', 'L2', 'rep', 't0', 't1', 'lo1', 'hi1', 'lo2', 'hi2', 'box_entries',
          'ours_ms', 'ours_err', 'hits', 'partial_hits', 'pieces1', 'pieces2', 'leaf_pieces', 'stitch', 'partial',
          'tucket_ms', 'tucket_err', 'tucket_hits', 'als_ms', 'als_err', 'ref_err', 'ref_ms']


def run_dataset(name):
    X = torch.tensor(np.load(osp.join(args.data_root, f'{name}.npy')))
    if args.T > 0:
        X = X[: args.T]
    X = X.contiguous()
    sizes = list(X.shape)
    ranks = [args.time_rank] + [max(1, M // args.static_rank_div) for M in sizes[1:]]
    qpath = osp.join(args.queries_dir or OUT, f'queries_{name}.json')
    if args.queries_dir:
        D = json.load(open(qpath))
        assert D['sizes'] == sizes and D['ranks'] == ranks, 'saved queries were made for another shape or ranks'
        log(f'== {name}: reusing {len(D["queries"])} queries from {qpath}')
    else:
        D = design(name, sizes, ranks)
        D.update(sizes = sizes, ranks = ranks, n_reps = args.n_reps, seed = args.seed)
        json.dump(D, open(qpath, 'w'))
    if not args.queries_dir or osp.abspath(args.queries_dir) != osp.abspath(OUT):
        json.dump(D, open(osp.join(OUT, f'queries_{name}.json'), 'w'))
    lv = D['levels']
    log(f'== {name}: shape {tuple(sizes)}, ranks {ranks}, levels Lt={lv[0]} L1={lv[1]} L2={lv[2]}, '
        f'{len(lv[0]) * len(lv[1]) * len(lv[2])} cells x {D["n_reps"]} endpoints = {len(D["queries"])} queries')

    trees, builds = {}, {}
    if 'ours' in METHODS:
        log('  building the multidimensional tree (query_dims = [1, 2]) ...')
        trees['ours'], s, ops = build(X, ranks, [1, 2])
        builds['ours'] = dict(build_s = s, **ops)
        log(f'  built in {s:.0f}s, {ops["stitch"]} STITCH, {ops["als"]} ALS')
    if 'tucket' in METHODS:
        trees['tucket'], s, ops = build(X, ranks, [])
        builds['tucket'] = dict(build_s = s, **ops)
        log(f'  original TUCKET built in {s:.1f}s')
    json.dump(builds, open(osp.join(OUT, f'builds_{name}.json'), 'w'), indent = 1)

    csv_path = osp.join(OUT, f'queries_{name}.csv')
    f = open(csv_path, 'w', newline = '')
    w = csv.DictWriter(f, fieldnames = FIELDS)
    w.writeheader()
    qs = D['queries']
    for q in qs[:3]:                                  # warm-up, not recorded
        if 'ours' in trees:
            trees['ours'].query_tucker(q['t0'], q['t1'], args.prune, ranges = {1: tuple(q['r1']), 2: tuple(q['r2'])})
    tic = now()
    for i, qid in enumerate(D['order']):
        q = qs[qid]
        (lo1, hi1), (lo2, hi2) = q['r1'], q['r2']
        Xq = X[q['t0']:q['t1'], lo1:hi1, lo2:hi2]
        norm = float(torch.linalg.norm(Xq.double().flatten()))
        row = dict(dataset = name, qid = qid, Lt = q['Lt'], L1 = q['L1'], L2 = q['L2'], rep = q['rep'], t0 = q['t0'],
                   t1 = q['t1'], lo1 = lo1, hi1 = hi1, lo2 = lo2, hi2 = hi2, box_entries = Xq.numel())
        if 'ours' in trees:
            tree = trees['ours']
            (tk, _, hits), ms = timed(lambda: tree.query_tucker(q['t0'], q['t1'], args.prune, ranges = {1: (lo1, hi1), 2: (lo2, hi2)}))
            st = tree.last_stats
            row.update(ours_ms = ms, ours_err = err64(Xq, tk) / norm, hits = hits, partial_hits = st['partial_hits'],
                       pieces1 = st['pieces'][1], pieces2 = st['pieces'][2], leaf_pieces = st['leaf_pieces'],
                       stitch = OP_COUNT['stitch'], partial = OP_COUNT['partial'])
        if 'tucket' in trees:
            base = trees['tucket']

            def tucket_query():
                tk, _, hits = base.query_tucker(q['t0'], q['t1'], args.prune)
                tk = tucker_partial(tk, lo1, hi1, qr = True, axis = 1)
                tk = tucker_partial(tk, lo2, hi2, qr = True, axis = 2)
                return tk, hits
            (tk, hits), ms = timed(tucket_query)
            row.update(tucket_ms = ms, tucket_err = err64(Xq, tk) / norm, tucket_hits = hits)
        if 'als' in METHODS:
            def als_query():
                return tucker_als(X[q['t0']:q['t1'], lo1:hi1, lo2:hi2], ranks, args.tol, args.maxiters)[0]
            tk, ms = timed(als_query)
            row.update(als_ms = ms, als_err = err64(Xq, tk) / norm)
        set_seed(args.seed + 1)
        t = now()
        tk, _ = tucker_als(Xq, ranks, args.als_tol, args.als_maxiters)
        row.update(ref_ms = 1e3 * (now() - t), ref_err = err64(Xq, tk) / norm)
        w.writerow(row)
        if (i + 1) % 100 == 0:
            f.flush()
            el = now() - tic
            log(f'    {i + 1}/{len(qs)} queries, {el:.0f}s elapsed, ~{el / (i + 1) * (len(qs) - i - 1):.0f}s left')
    f.close()
    del trees
    analyse(name)


def analyse(name):
    import pandas as pd
    df = pd.read_csv(osp.join(OUT, f'queries_{name}.csv'))
    keys = ['Lt', 'L1', 'L2']
    g = df.groupby(keys)
    cells = [tuple(int(v) for v in k) for k in g.groups.keys()]
    res = dict(dataset = name, n_cells = len(cells), n_queries = len(df), fits = {})
    targets = {}
    for m in METHODS:
        targets[f'{m}_ms'] = g[f'{m}_ms'].mean().values
    if 'ours' in METHODS:
        targets['leaf_pieces'] = g['leaf_pieces'].mean().values
    for tname, y in targets.items():
        fits, best = fit_models(cells, y)
        res['fits'][tname] = dict(best = best, models = {k: {kk: vv for kk, vv in v.items() if kk != 'pred'} for k, v in fits.items()})
        res['fits'][tname]['_pred'] = {k: v['pred'] for k, v in fits.items()}
    for m in METHODS:
        e = df[f'{m}_err'] / df['ref_err'] - 1
        res[f'{m}_eq29_median'] = float(e[df['ref_err'] > 1e-6].median())
        res[f'{m}_ms_median'] = float(df[f'{m}_ms'].median())
    res['cells'] = cells
    if 'ours' in METHODS:                        # model-free checks of the structure
        cm = g.agg(leaf = ('leaf_pieces', 'mean'), H = ('hits', 'mean'), s1 = ('pieces1', 'mean'), s2 = ('pieces2', 'mean'))
        res['identity_leaf_eq_H_s1_s2'] = float((df['leaf_pieces'] == df['hits'] * df['pieces1'] * df['pieces2']).mean())
        res['identity_leaf_eq_H_plus_s1_plus_s2'] = float((df['leaf_pieces'] == df['hits'] + df['pieces1'] + df['pieces2']).mean())
        res['corr_cellmean_leaf_vs_prod_of_means'] = float(np.corrcoef(cm['leaf'], cm['H'] * cm['s1'] * cm['s2'])[0, 1])
        res['corr_cellmean_time_vs_leaf'] = float(np.corrcoef(g['ours_ms'].mean().values, cm['leaf'].values)[0, 1])
        per_mode = {}                            # one-dimensional: mean count per level of each mode alone
        for col, key in (('hits', 'Lt'), ('pieces1', 'L1'), ('pieces2', 'L2')):
            s_ = df.groupby(key)[col].mean()
            Ls, ys = s_.index.values.astype(float), s_.values
            fits1 = {}
            for mname, feat in (('const', []), ('log L', [np.log2(Ls)]), ('sqrt L', [np.sqrt(Ls)]), ('L', [Ls])):
                A = np.column_stack([np.ones(len(Ls))] + feat)
                c_, *_ = np.linalg.lstsq(A, ys, rcond = None)
                r_ = ys - A @ c_
                fits1[mname] = dict(coef = c_.tolist(), rss = float(r_ @ r_))
            per_mode[key] = dict(levels = Ls.astype(int).tolist(), mean_count = ys.tolist(), fits = fits1)
        res['per_mode_counts'] = per_mode
    agg = g.agg({**{f'{m}_ms': ['mean', 'std'] for m in METHODS}, **({'leaf_pieces': ['mean', 'std'], 'pieces1': 'mean', 'pieces2': 'mean', 'hits': 'mean'} if 'ours' in METHODS else {})})
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg.to_csv(osp.join(OUT, f'cells_{name}.csv'))
    json.dump(res, open(osp.join(OUT, f'fits_{name}.json'), 'w'), indent = 1)
    lines = [f'{name}: {len(df)} queries in {len(cells)} cells']
    for tname, fr in res['fits'].items():
        ms = fr['models']
        order = sorted(ms, key = lambda k: ms[k]['loocv_rmse'])
        lines.append(f'  {tname}: best = {fr["best"]}')
        for k in order:
            lines.append(f'      {k:26s} LOOCV rel. RMSE {100 * ms[k]["loocv_rel"]:7.2f}%   R2 {ms[k]["r2"]:.4f}   BIC {ms[k]["bic"]:9.1f}')
    if 'ours' in METHODS:
        lines.append(f'  per query: leaf pieces == hits*s1*s2 for {100 * res["identity_leaf_eq_H_s1_s2"]:.1f}% of queries, '
                     f'== hits+s1+s2 for {100 * res["identity_leaf_eq_H_plus_s1_plus_s2"]:.1f}%')
        lines.append(f'  cell means: corr(leaf, mean H * mean s1 * mean s2) = {res["corr_cellmean_leaf_vs_prod_of_means"]:.4f}, '
                     f'corr(time, leaf) = {res["corr_cellmean_time_vs_leaf"]:.4f}')
        for key, pm in res['per_mode_counts'].items():
            best1 = min(pm['fits'], key = lambda k: pm['fits'][k]['rss'])
            lines.append(f'  mean count vs {key}: ' + ', '.join(f'{L}:{c:.2f}' for L, c in zip(pm['levels'], pm['mean_count']))
                         + f'   (1-D best of const/log/sqrt/linear: {best1})')
    for m in METHODS:
        lines.append(f'  {m}: median query {res[f"{m}_ms_median"]:.2f} ms, Eq. 29 median {100 * res[f"{m}_eq29_median"]:.2f}%')
    txt = '\n'.join(lines)
    log(txt)
    with open(osp.join(OUT, 'summary.txt'), 'a') as fo:
        fo.write(txt + '\n\n')
    plot(name, df, agg, res)


def plot(name, df, agg, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6})
    INK, INK2 = '#0b0b0b', '#52514e'
    CAT = {'ours': '#2a78d6', 'tucket': '#eb6834', 'als': '#1baf7a'}
    LABEL = {'ours': 'ours (multidim. tree)', 'tucket': 'TUCKET + slicing', 'als': 'Tucker-ALS on the box'}
    a = agg.reset_index()
    Lts, L2s = sorted(a['Lt'].unique()), sorted(a['L2'].unique())
    blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(L2s)))

    # 1) the p-dimensional table of the tree: time vs L1, one line per L2, one panel per Lt
    if 'ours' in METHODS:
        for col, ylabel, fname in (('ours_ms_mean', 'mean query time (ms)', 'time'), ('leaf_pieces_mean', 'mean leaf pieces', 'pieces')):
            fig, axs = plt.subplots(1, len(Lts), figsize = (2.6 * len(Lts) + 1, 3), sharey = True, squeeze = False)
            for ax, Lt in zip(axs[0], Lts):
                for c, L2 in zip(blues, L2s):
                    s = a[(a['Lt'] == Lt) & (a['L2'] == L2)].sort_values('L1')
                    ax.plot(s['L1'], s[col], '-o', color = c, lw = 2, ms = 4, label = f'L2={L2}')
                ax.set_xscale('log', base = 2)
                L1s = sorted(a['L1'].unique())
                ax.set_xticks(L1s)
                ax.set_xticklabels([str(int(v)) for v in L1s])
                ax.minorticks_off()
                ax.set_title(f'Lt = {Lt}', color = INK)
                ax.set_xlabel('L1 (log scale)', color = INK2)
            axs[0][0].set_ylabel(ylabel, color = INK2)
            axs[0][-1].legend(fontsize = 7, frameon = False, loc = 'upper left', bbox_to_anchor = (1.0, 1.0))
            best = res['fits']['ours_ms' if fname == 'time' else 'leaf_pieces']['best']
            fig.suptitle(f'{name}: {ylabel} over the full Lt x L1 x L2 table (mean over endpoints; best fit: {best})',
                         fontsize = 9, color = INK)
            fig.tight_layout()
            fig.savefig(osp.join(OUT, f'fig_table_{fname}_{name}.png'), dpi = 160, bbox_inches = 'tight')
            plt.close(fig)

    # 2) model comparison per method (LOOCV relative RMSE)
    tn = [f'{m}_ms' for m in METHODS] + (['leaf_pieces'] if 'ours' in METHODS else [])
    fig, axs = plt.subplots(1, len(tn), figsize = (3.4 * len(tn), 3.2), squeeze = False)
    for ax, t in zip(axs[0], tn):
        ms = res['fits'][t]['models']
        names = list(ms)
        vals = [100 * ms[k]['loocv_rel'] for k in names]
        best = res['fits'][t]['best']
        ax.barh(range(len(names)), vals, color = ['#2a78d6' if k == best else '#c3c2b7' for k in names], height = 0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize = 7)
        ax.invert_yaxis()
        ax.set_xlim(0.9 * min(vals), 1.02 * max(vals))
        ax.set_xlabel('LOOCV relative RMSE (%)  - lower is better', color = INK2, fontsize = 7)
        ax.set_title(t.replace('_ms', ' time'), color = INK)
        ax.grid(axis = 'y', visible = False)
    fig.suptitle(f'{name}: which complexity model explains the cell means (best in blue)', fontsize = 9, color = INK)
    fig.tight_layout()
    fig.savefig(osp.join(OUT, f'fig_models_{name}.png'), dpi = 160, bbox_inches = 'tight')
    plt.close(fig)

    # 3) methods on the same queries: time and Eq. 29 vs box size
    fig, axs = plt.subplots(1, 2, figsize = (8.5, 3.2))
    vol = df['Lt'] * df['L1'] * df['L2']
    for m in METHODS:
        gm = df.assign(vol = vol).groupby('vol')
        axs[0].plot(gm[f'{m}_ms'].mean().index, gm[f'{m}_ms'].mean().values, 'o', ms = 3, color = CAT[m], label = LABEL[m], alpha = 0.8)
        e = (df[f'{m}_err'] / df['ref_err'] - 1)[df['ref_err'] > 1e-6]
        ge = e.groupby(vol[df['ref_err'] > 1e-6]).median()
        axs[1].plot(ge.index, 100 * ge.values, 'o', ms = 3, color = CAT[m], label = LABEL[m], alpha = 0.8)
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlabel('box size Lt*L1*L2 (log scale)', color = INK2)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('mean query time (ms, log scale)', color = INK2)
    axs[1].set_ylabel('median Eq. 29: err / err_ref - 1 (%)', color = INK2)
    axs[1].set_yscale('symlog', linthresh = 1)
    axs[0].legend(fontsize = 7, frameon = False)
    fig.suptitle(f'{name}: all methods on the same queries', fontsize = 9, color = INK)
    fig.tight_layout()
    fig.savefig(osp.join(OUT, f'fig_methods_{name}.png'), dpi = 160, bbox_inches = 'tight')
    plt.close(fig)


def replot(name):
    """redraw the figures of a finished data set from its saved CSV and fits"""
    import pandas as pd
    df = pd.read_csv(osp.join(OUT, f'queries_{name}.csv'))
    agg = pd.read_csv(osp.join(OUT, f'cells_{name}.csv')).set_index(['Lt', 'L1', 'L2'])
    plot(name, df, agg, json.load(open(osp.join(OUT, f'fits_{name}.json'))))


# ------------------------------------------------------------------ task error_k
TREES = [(), (1,), (2,), (1, 2)]


def run_error_k(name):
    """Reconstruction error vs the number k of query modes, on the same query design as the complexity task.

    The base queries are the complexity design (every cell of the Lt x L1 x L2 table, --n_reps random
    endpoints, independent per mode; saved once to queries_<name>.json and reused). Every base query is
    answered by four trees, query_dims = [], [1], [2], [1, 2] (k = 0, 1, 1, 2), in two families:
      same_box    the static modes in full: the SAME box for every k, only the tree changes
      restricted  the query restricts exactly the k modes that have a tree, with the base query's ranges
                  (k = 0 has no static restriction, so it equals same_box)
    Each box is also decomposed by Tucker-ALS directly (--als_tol, --als_maxiters; one run per distinct box),
    and both errors are relative: ||X_q - Y|| / ||X_q||."""
    X = torch.tensor(np.load(osp.join(args.data_root, f'{name}.npy')))
    if args.T > 0:
        X = X[: args.T]
    X = X.contiguous()
    sizes = list(X.shape)
    ranks = [args.time_rank] + [max(1, M // args.static_rank_div) for M in sizes[1:]]
    qpath = osp.join(args.queries_dir or OUT, f'queries_{name}.json')
    if args.queries_dir:
        D = json.load(open(qpath))
        assert D['sizes'] == sizes and D['ranks'] == ranks, 'saved queries were made for another shape or ranks'
    else:
        D = design(name, sizes, ranks)
        D.update(sizes = sizes, ranks = ranks, n_reps = args.n_reps, seed = args.seed)
        json.dump(D, open(qpath, 'w'))
    qs = D['queries']
    lv = D['levels']
    log(f'== {name} (error vs k): shape {tuple(sizes)}, ranks {ranks}, levels Lt={lv[0]} L1={lv[1]} L2={lv[2]}, '
        f'{len(qs)} base queries, trees k = 0, 1, 1, 2')
    als_cache = {}
    rows = []
    for S in TREES:
        log(f'  tree query_dims = {list(S)}: building ...')
        tree, bs, ops = build(X, ranks, list(S))
        log(f'  built in {bs:.0f}s ({ops["stitch"]} STITCH)')
        fams = ['same_box'] if not S else ['same_box', 'restricted']
        for i, qid in enumerate(D['order']):
            q = qs[qid]
            for fam in fams:
                ranges = {} if fam == 'same_box' else {m: tuple(q[f'r{m}']) for m in S}
                box = tuple(ranges.get(m, (0, sizes[m])) for m in (1, 2))
                Xq = X[q['t0']:q['t1'], box[0][0]:box[0][1], box[1][0]:box[1][1]]
                norm = float(torch.linalg.norm(Xq.double().flatten()))
                if norm == 0.:
                    continue
                set_seed(args.seed + 1)
                tic = now()
                tk, _, hits = tree.query_tucker(q['t0'], q['t1'], args.prune, ranges = ranges)
                ms = 1e3 * (now() - tic)
                key = (q['t0'], q['t1'], box)
                if key not in als_cache:
                    set_seed(args.seed + 1)
                    tka, _ = tucker_als(Xq, ranks, args.als_tol, args.als_maxiters)
                    als_cache[key] = err64(Xq, tka) / norm
                rows.append(dict(dataset = name, tree = '-'.join(map(str, S)) or 'none', k = len(S), family = fam,
                                 qid = qid, Lt = q['Lt'], L1 = q['L1'], L2 = q['L2'], rep = q['rep'], t0 = q['t0'], t1 = q['t1'],
                                 box = ';'.join(f'{a}:{b}' for a, b in box), box_entries = Xq.numel(), hits = hits,
                                 query_ms = ms, err_tree = err64(Xq, tk) / norm, err_als = als_cache[key]))
            if (i + 1) % 500 == 0:
                log(f'    {i + 1}/{len(qs)} base queries')
        del tree
    import pandas as pd
    df = pd.DataFrame(rows)
    df['eq29'] = df['err_tree'] / df['err_als'] - 1
    df.to_csv(osp.join(OUT, f'errk_{name}.csv'), index = False)
    tab = df.groupby(['family', 'k']).agg(n = ('err_tree', 'size'), err_tree = ('err_tree', 'median'),
                                          err_als = ('err_als', 'median'), eq29 = ('eq29', 'median'))
    txt = f'{name}: reconstruction error vs k (medians; eq29 = err_tree / err_als - 1)\n' + tab.round(4).to_string()
    log(txt)
    with open(osp.join(OUT, 'summary.txt'), 'a') as fo:
        fo.write(txt + '\n\n')
    plot_error_k()


def plot_error_k():
    """one figure with every data set found in OUT (errk_<name>.csv)"""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for n in ALL if osp.exists(osp.join(OUT, f'errk_{n}.csv'))]
    if not names:
        return
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6})
    BLUE, GREEN, INK, INK2 = '#2a78d6', '#1baf7a', '#0b0b0b', '#52514e'
    fig, axs = plt.subplots(2, len(names), figsize = (max(4.6, 3.3 * len(names)), 6.0), squeeze = False)
    for j, name in enumerate(names):
        df = pd.read_csv(osp.join(OUT, f'errk_{name}.csv'))
        k0 = df[df['k'] == 0]
        for fam, ls, flabel in (('same_box', '-', 'same box for every k'), ('restricted', '--', 'box restricted on the k tree modes')):
            s = pd.concat([k0, df[(df['k'] > 0) & (df['family'] == fam)]])   # k = 0 has no static restriction
            for col, color, mlabel, dx in (('err_tree', BLUE, 'ours', -0.04), ('err_als', GREEN, 'Tucker-ALS', 0.04)):
                g = s.groupby('k')[col]
                ks = np.array(sorted(g.groups))
                med, q1, q3 = g.median().loc[ks], g.quantile(0.25).loc[ks], g.quantile(0.75).loc[ks]
                axs[0][j].errorbar(ks + dx, med, yerr = [med - q1, q3 - med], fmt = 'o' + ls, color = color, lw = 2,
                                   ms = 5, capsize = 2, elinewidth = 1, label = f'{mlabel}, {flabel}')
            g = s.groupby('k')['eq29']
            ks = np.array(sorted(g.groups))
            axs[1][j].plot(ks, 100 * g.median().loc[ks], 'o' + ls, color = BLUE, lw = 2, ms = 5, label = flabel)
        axs[0][j].set_title(name, color = INK)
        for i in range(2):
            axs[i][j].set_xticks([0, 1, 2])
            axs[i][j].set_xlabel('number of query modes k', color = INK2)
        axs[1][j].set_ylim(bottom = 0)
    axs[0][0].set_ylabel('relative reconstruction error\n(median, IQR)', color = INK2)
    axs[1][0].set_ylabel('median err_tree / err_ALS - 1 (%)', color = INK2)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize = 7, frameon = False, loc = 'upper center', ncol = 2, bbox_to_anchor = (0.5, 0.0))
    fig.suptitle('Reconstruction error vs number of query modes, multidimensional tree vs Tucker-ALS on the same boxes',
                 fontsize = 9, color = INK)
    fig.tight_layout()
    fig.savefig(osp.join(OUT, 'fig_error_vs_k.png'), dpi = 160, bbox_inches = 'tight')
    plt.close(fig)


if __name__ == '__main__':
    json.dump(vars(args), open(osp.join(OUT, 'config_' + '_'.join(args.datasets) + '.json'), 'w'), indent = 1)
    for name in args.datasets:
        run_dataset(name) if args.task == 'complexity' else run_error_k(name)
    log(f'saved results to {OUT}/')
