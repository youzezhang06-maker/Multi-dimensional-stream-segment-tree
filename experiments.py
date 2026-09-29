"""
Experiments for the multidimensional TucketTree (test_tucket.py).

The data set fixes the total number of modes p. We vary k = the number of
non-temporal modes that support range queries (query_dims), from 0 (original
TUCKET) to p - 1. A tree is built ONLY on the modes in query_dims; the other
non-temporal modes get no tree and every query takes them in full. For each k,
every subset of k non-temporal modes is tried as query_dims, and every query
restricts time plus every mode in query_dims.

Measured for every setting:
  accuracy  reconstruction error of the tree's answer vs. Tucker-ALS run directly
            on the exact queried sub-box (same ranks, same seed) -- the two should
            be close for every k and every data set (correctness check)
  cost      build time, number of ALS / STITCH calls during the build (compared with
            the formula (T-1)*P + T*(P - prod M_i), P = prod_{i in query_dims}(2 M_i - 1)),
            stored Tucker decompositions and floats, query time, STITCH calls per query

A tree on many modes stores P Tucker decompositions per temporal node, so the data
is cropped first (--T time steps, first --static entries of every non-temporal mode).

Usage (from the TUCKET root, datasets in inputs/):
    python experiments.py --device cuda:0
    python experiments.py --datasets AirQuality Traffic --T 128 --static 6 --ranks 4,3,3
Outputs in --out_dir:
    exp_error.png   error vs. k (tree vs. Tucker-ALS), one panel per data set
    exp_cost.png    build STITCH calls (measured vs. formula), storage, query time vs. k
    exp_queries.csv one row per query;  exp_settings.csv one row per (data set, query_dims)
"""
import argparse
import csv
import itertools
import os
import os.path as osp
import time

import numpy as np
import torch

from inc.utils import set_seed
from inc.tucker import *
from test_tucket import TucketTree

ap = argparse.ArgumentParser()
ap.add_argument('--datasets', nargs = '+', default = ['AirQuality', 'Traffic', 'USStock', 'KRStock'])
ap.add_argument('--data_root', default = 'inputs')
ap.add_argument('--T', type = int, default = 64, help = 'number of time steps to use')
ap.add_argument('--static', type = int, default = 4, help = 'max size kept on every non-temporal mode')
ap.add_argument('--ranks', type = lambda s: [int(x) for x in s.split(',')], default = [3, 3, 2])
ap.add_argument('--tol', type = float, default = 1e-2)
ap.add_argument('--maxiters', type = int, default = 8)
ap.add_argument('--prune', type = float, default = 0.7)
ap.add_argument('--n_queries', type = int, default = 6, help = 'queries per setting')
ap.add_argument('--device', default = 'cpu')
ap.add_argument('--seed', type = int, default = 998244353)
ap.add_argument('--out_dir', default = 'outputs')
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok = True)
CUDA = str(args.device).startswith('cuda')


def now():
    if CUDA:
        torch.cuda.synchronize()
    return time.perf_counter()


def load(name):
    X = np.load(osp.join(args.data_root, f'{name}.npy')).astype(np.float32)
    X = X[(slice(0, args.T),) + tuple(slice(0, args.static) for _ in range(X.ndim - 1))]
    return torch.tensor(np.ascontiguousarray(X), device = args.device)


def rel_err(X, tucker):
    Xhat = tensor_mats_mul(tucker.G, A_dim_list = [(Up, p) for p, Up in enumerate(tucker.U)])
    return (tensor_norm(X - Xhat) / tensor_norm(X)).item()


def rand_time(rng, T):
    length = int(rng.integers(max(2, T // 12), max(3, T // 3)))
    t0 = int(rng.integers(0, T - length + 1))
    return t0, t0 + length


def rand_range(rng, size):
    """random [lo, hi) of length >= 2 (length-1 boxes are almost trivially reconstructable)"""
    lo = int(rng.integers(0, size - 1))
    return lo, lo + int(rng.integers(2, size - lo + 1))


def formula_build_stitches(T, sizes, query_dims):
    """STITCH calls to build the tree: (T-1) temporal merges x P, plus (P - prod M_i) per time step"""
    P = int(np.prod([2 * sizes[m] - 1 for m in query_dims])) if query_dims else 1
    M = int(np.prod([sizes[m] for m in query_dims])) if query_dims else 1
    return (T - 1) * P + T * (P - M)


def run_setting(X, query_dims, rng):
    """build a tree on query_dims, answer n_queries random queries; return (setting row, query rows)"""
    T, sizes = X.size(0), list(X.shape)
    set_seed(args.seed)
    reset_op_count()
    tic = now()
    tree = TucketTree(args.ranks, args.tol, args.maxiters, alloc = T, query_dims = query_dims)
    for t in range(T):
        tree.append(X[t])
    build_time = now() - tic
    build_ops = dict(OP_COUNT)
    n_tuckers, n_floats = tree.storage()
    qrows = []
    for _ in range(args.n_queries):
        t0, t1 = rand_time(rng, T)
        ranges = {m: rand_range(rng, sizes[m]) for m in query_dims}
        Xq = X[(slice(t0, t1),) + tuple(slice(*ranges[m]) if m in ranges else slice(None) for m in range(1, X.dim()))]
        set_seed(args.seed + 1)
        reset_op_count()
        tic = now()
        tucker, _, hits = tree.query_tucker(t0, t1, args.prune, ranges = ranges)
        q_time = now() - tic
        q_ops = dict(OP_COUNT)
        e_tree = rel_err(Xq, tucker)
        set_seed(args.seed + 1)
        e_als = rel_err(Xq, tucker_als(Xq, args.ranks, args.tol, args.maxiters)[0])
        qrows.append(dict(t0 = t0, t1 = t1, ranges = ranges, hits = hits, time = q_time,
                          stitch = q_ops['stitch'], partial = q_ops['partial'], err_tree = e_tree, err_als = e_als))
    setting = dict(query_dims = list(query_dims), k = len(query_dims), build_time = build_time,
                   build_als = build_ops['als'], build_stitch = build_ops['stitch'],
                   formula_stitch = formula_build_stitches(T, sizes, query_dims),
                   n_tuckers = n_tuckers, n_floats = n_floats, raw_floats = X.numel(),
                   query_time = np.mean([q['time'] for q in qrows]), query_stitch = np.mean([q['stitch'] for q in qrows]),
                   err_tree = np.mean([q['err_tree'] for q in qrows]), err_als = np.mean([q['err_als'] for q in qrows]))
    return setting, qrows


# ------------------------------------------------------------------ run
settings, queries = {}, []
for name in args.datasets:
    X = load(name)
    modes = list(range(1, X.dim()))
    print(f'== {name}: using shape {tuple(X.shape)}')
    rng = np.random.default_rng(0)
    settings[name] = []
    for k in range(len(modes) + 1):
        for query_dims in itertools.combinations(modes, k):
            s, qrows = run_setting(X, list(query_dims), rng)
            settings[name].append(s)
            queries += [dict(dataset = name, k = k, query_dims = list(query_dims), **q) for q in qrows]
            print(f'  query_dims={list(query_dims)!s:9s} err tree {s["err_tree"]:.4f} / Tucker-ALS {s["err_als"]:.4f}   '
                  f'build {s["build_time"]:6.2f}s, STITCH {s["build_stitch"]} (formula {s["formula_stitch"]})   '
                  f'storage {s["n_floats"] / s["raw_floats"]:6.1f}x raw   query {s["query_time"] * 1e3:6.1f} ms, '
                  f'{s["query_stitch"]:.1f} STITCH')

# ------------------------------------------------------------------ save
with open(osp.join(args.out_dir, 'exp_queries.csv'), 'w', newline = '') as f:
    w = csv.DictWriter(f, fieldnames = list(queries[0].keys()))
    w.writeheader()
    w.writerows(queries)
with open(osp.join(args.out_dir, 'exp_settings.csv'), 'w', newline = '') as f:
    rows = [dict(dataset = name, **s) for name, ss in settings.items() for s in ss]
    w = csv.DictWriter(f, fieldnames = list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ------------------------------------------------------------------ plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE = '#2a78d6', '#eb6834'


def by_k(name, key):
    """mean of `key` over all settings with the same k (all subsets of k modes)"""
    ks = sorted({s['k'] for s in settings[name]})
    return ks, [np.mean([s[key] for s in settings[name] if s['k'] == k]) for k in ks]


def style(ax, xs, xlabel):
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel)
    ax.grid(axis = 'y', alpha = 0.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


names = list(settings)
xlabel = '# query dims (modes with a tree)'

# error vs k: per-query mean +- std, tree vs Tucker-ALS
fig, axes = plt.subplots(1, len(names), figsize = (3.8 * len(names), 3.4), sharey = True, squeeze = False)
for ax, name in zip(axes[0], names):
    ks = sorted({q['k'] for q in queries if q['dataset'] == name})
    for key, label, color, marker, ls in [('err_tree', 'tree', BLUE, 'o', '-'),
                                          ('err_als', 'Tucker-ALS on exact sub-box', ORANGE, 's', '--')]:
        vals = [[q[key] for q in queries if q['dataset'] == name and q['k'] == k] for k in ks]
        ax.errorbar(ks, [np.mean(v) for v in vals], yerr = [np.std(v) for v in vals], color = color, marker = marker,
                    linestyle = ls, linewidth = 2, capsize = 3, label = label)
    ax.set_title(name)
    style(ax, ks, xlabel)
axes[0][0].set_ylabel('relative reconstruction error')
axes[0][0].legend(fontsize = 8)
fig.suptitle('Reconstruction error vs. number of query dimensions')
fig.tight_layout()
fig.savefig(osp.join(args.out_dir, 'exp_error.png'), dpi = 150)
plt.close(fig)

# cost vs k: one row per metric, one column per data set
metrics = [('build_stitch', 'build STITCH calls', True), ('storage_ratio', 'storage / raw data', True),
           ('query_time_ms', 'query time (ms)', False)]
for ss in settings.values():
    for s in ss:
        s['storage_ratio'] = s['n_floats'] / s['raw_floats']
        s['query_time_ms'] = s['query_time'] * 1e3
fig, axes = plt.subplots(len(metrics), len(names), figsize = (3.8 * len(names), 2.8 * len(metrics)), squeeze = False)
for j, name in enumerate(names):
    for i, (key, label, log) in enumerate(metrics):
        ax = axes[i][j]
        ks, ys = by_k(name, key)
        ax.plot(ks, ys, color = BLUE, marker = 'o', linewidth = 2, label = 'measured')
        if key == 'build_stitch':
            ax.plot(ks, by_k(name, 'formula_stitch')[1], color = ORANGE, marker = 's', linestyle = '--',
                    linewidth = 2, label = 'formula (T-1)P + T(P - prod M)')
            ax.legend(fontsize = 7)
        if log:
            ax.set_yscale('log')
        if i == 0:
            ax.set_title(name)
        if j == 0:
            ax.set_ylabel(label)
        style(ax, ks, xlabel if i == len(metrics) - 1 else '')
fig.suptitle('Cost vs. number of query dimensions (mean over all subsets of k modes)')
fig.tight_layout()
fig.savefig(osp.join(args.out_dir, 'exp_cost.png'), dpi = 150)
plt.close(fig)
print(f'saved exp_error.png, exp_cost.png, exp_queries.csv, exp_settings.csv to {args.out_dir}/')
