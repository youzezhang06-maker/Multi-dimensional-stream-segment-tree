"""
Experiments for the multidimensional TucketTree (test_tucket.py).

Design rules (each fixes an issue found in the previous version; numbers refer to the issue list):
  * Metric (1, 2): the paper's Eq. 29, err_tree / err_ALS - 1, where err = ||X_q - Y||_F (float64) and
    Y_ALS is Tucker-ALS run directly on the exact queried box (the most accurate method; the reference).
    Queries whose ALS error is ~0 (box exactly representable at the given ranks) are flagged "trivial"
    and excluded from Eq. 29, as are all-zero boxes.
  * Settings first, then a fixed number of repetitions per setting (5, 6, 7). All queries are generated
    ONCE per data set, before any tree is built, and saved to queries_<dataset>.json. The same time
    windows and the same relative positions are reused across all settings (paired design).
  * Ranges are designed per data set (8): ranks are fixed first (per mode), then every queried range
    satisfies rank < length < mode size; lengths come from fixed proportional levels; then t0, t1.
  * Control variables (4): in the "error vs k" experiment the queried static volume is held fixed:
    with k restricted modes each one keeps a fraction rho^(1/k) of its mode, so the box keeps a fraction
    rho of the static entries for every k. A second query family ("control_full") asks the SAME box
    (all static modes in full) on every tree, so only the nesting depth changes.
  * Tucker-ALS baseline (9) has its own convergence settings (tight tol, many iterations), separate from
    the tree's, and its error is reported so that its flatness across k can be checked.
  * Streaming (10): queries are also issued DURING the build, at checkpoints that are not powers of two,
    on windows ending at the newest slice (the right spine of the tree is incomplete there).
  * Full time span by default (11); the static modes are cropped (a tree on a mode of size M stores 2M-1
    nodes per temporal node) after dropping entities without data, and everything is recorded in
    config.json. If the estimated memory exceeds --max_gb, T is reduced and this is logged.
  * Query length as the variable (12): experiment "length" sweeps every valid length on each mode and
    records the number of canonical pieces per mode (bound 2*floor(log2 L)+2) and the query cost.
  * More modes / deeper nesting (13): a synthetic low-rank tensor with 4 equal static modes; the number of
    trees k grows while the data stay the same.
  * Implementation (14, 15 -- see test_tucket.py): invalid ranges raise instead of being truncated; the
    stitch fit/stopping rule uses the exact norm of the queried box.
  * Build STITCH count (16): checked against the general formula (T - popcount(T)) * P + T * (P - prod M),
    P = prod_{i in S} (2 M_i - 1), valid for any T (not only powers of two).
  * Figures carry the corresponding time / space complexity (3).

Experiments (--experiments, comma separated):
  k       error and cost vs the number k of query modes (tree on exactly those modes), per data set
  length  query cost vs range length L on each mode (tree on all queryable modes, aligned time window)
  stream  queries during the build, at checkpoints (run together with k)
  synth   synthetic tensor, nesting depth k = 0..--syn_max_k

Usage (from the TUCKET root, data sets in inputs/):
  python experiments.py --quick                      # small smoke test, minutes
  python experiments.py                              # full time span, all data sets (hours)
  python experiments.py --datasets Traffic --static Traffic=32,32 --experiments k,length

Outputs (a new folder under results/, never overwritten):
  config.json            all arguments, per-data-set shapes, kept entity indices, ranks, length levels, T used
  queries_<name>.json    every generated query (setting, repetition, t0, t1, ranges)
  builds.csv             one row per (data set, query modes): build time, per-append latency, ALS/STITCH counts,
                         formula check, stored floats with and without zero padding
  queries.csv            one row per query: experiment, family, setting, box, hits, pieces per mode, STITCH and
                         partial calls, query time (median of --time_reps), errors, Eq. 29, ALS time, flags
  fig_*.png              figures (see the plot section)
  summary.txt            short per-data-set summary
"""
import argparse
import csv
import itertools
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
from test_tucket import TucketTree, _static_decompose_nodes

ALL_DATASETS = ['AirQuality', 'Traffic', 'USStock', 'KRStock']
DEFAULT_STATIC = dict(AirQuality = [8, 6],  # AirQuality has 21000 time steps: 16 would take ~3x longer
                       Traffic = [16, 16], USStock = [16, 16], KRStock = [16, 16])


def parse_static(s):
    """'AirQuality=16,6;Traffic=32,32' -> {'AirQuality': [16, 6], 'Traffic': [32, 32]}"""
    out = dict(DEFAULT_STATIC)
    for part in filter(None, (s or '').split(';')):
        name, sizes = part.split('=')
        out[name.strip()] = [int(x) for x in sizes.split(',')]
    return out


def floats(s):
    return [float(x) for x in s.split(',') if x]


ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
ap.add_argument('--datasets', nargs = '+', default = ALL_DATASETS)
ap.add_argument('--data_root', default = 'inputs')
ap.add_argument('--out_root', default = 'results')
ap.add_argument('--experiments', default = 'k,length,stream,synth')
ap.add_argument('--T', type = int, default = 0, help = 'time steps to use; 0 = the full time span')
ap.add_argument('--static', type = parse_static, default = parse_static(''),
                help = 'kept size of each non-temporal mode per data set, e.g. "AirQuality=16,6;Traffic=16,16"')
ap.add_argument('--min_coverage', type = float, default = 0.95,
                help = 'keep only entities (e.g. companies) whose fraction of nonzero entries is at least this')
ap.add_argument('--standardize', action = 'store_true', help = 'z-score every entity of the last mode (off by default)')
ap.add_argument('--time_rank', type = int, default = 5)
ap.add_argument('--static_rank_div', type = int, default = 4, help = 'static rank of a mode of size M = max(1, M // div)')
ap.add_argument('--ranks', default = '', help = 'override, e.g. "AirQuality=5,4,1;Traffic=5,4,4" (time rank first)')
ap.add_argument('--tol', type = float, default = 1e-2, help = 'ALS/STITCH tolerance inside the tree (paper: 0.01)')
ap.add_argument('--maxiters', type = int, default = 20, help = 'ALS/STITCH iterations inside the tree (paper: 20)')
ap.add_argument('--als_tol', type = float, default = 1e-6, help = 'tolerance of the Tucker-ALS reference')
ap.add_argument('--als_maxiters', type = int, default = 200, help = 'iterations of the Tucker-ALS reference')
ap.add_argument('--prune', type = float, default = 0.7)
ap.add_argument('--time_fracs', type = floats, default = floats('0.01,0.1,0.5'), help = 'time window lengths as fractions of T')
ap.add_argument('--rho', type = floats, default = floats('0.25,0.5'), help = 'fraction of static entries kept by a query (k experiment)')
ap.add_argument('--len_grid_fracs', type = floats, default = floats('0.25,0.5,0.75'), help = '2-D grid of the length experiment')
ap.add_argument('--checkpoints', type = floats, default = floats('0.3,0.55,0.8'), help = 'stream: checkpoints as fractions of T')
ap.add_argument('--n_reps', type = int, default = 20, help = 'repetitions (queries) per setting')
ap.add_argument('--time_reps', type = int, default = 3, help = 'timed runs per query (median reported)')
ap.add_argument('--trivial', type = float, default = 1e-6, help = 'ALS relative error below which a box counts as trivial')
ap.add_argument('--max_gb', type = float, default = 4.0, help = 'reduce T if the estimated tree memory exceeds this')
ap.add_argument('--syn_T', type = int, default = 300)
ap.add_argument('--syn_M', type = int, default = 8)
ap.add_argument('--syn_modes', type = int, default = 4, help = 'number of static modes of the synthetic tensor')
ap.add_argument('--syn_max_k', type = int, default = 3)
ap.add_argument('--syn_noise', type = float, default = 0.1)
ap.add_argument('--device', default = 'cpu')
ap.add_argument('--seed', type = int, default = 998244353)
ap.add_argument('--quick', action = 'store_true', help = 'small smoke test: T=200, static 8, 4 reps, 1 timed run')
args = ap.parse_args()
if args.quick:
    args.T = args.T or 200
    args.static = {k: [min(8, m) for m in v] for k, v in args.static.items()}
    args.n_reps, args.time_reps, args.syn_T, args.syn_M, args.syn_max_k = 4, 1, 100, 4, 2
EXPS = set(args.experiments.split(','))
RANK_OVERRIDE = parse_static(args.ranks) if args.ranks else {}
CUDA = str(args.device).startswith('cuda')
DTYPE = torch.float32

stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
OUT = osp.join(args.out_root, f'exp-{stamp}' + ('-quick' if args.quick else ''))
os.makedirs(OUT, exist_ok = False)


def log(msg):
    print(msg, flush = True)
    with open(osp.join(OUT, 'log.txt'), 'a') as f:
        f.write(msg + '\n')


def now():
    if CUDA:
        torch.cuda.synchronize()
    return time.perf_counter()


def popcount(n):
    return bin(n).count('1')


def formula_build(T, sizes, S):
    """(STITCH, ALS) calls to build the tree for any T: a temporal node is merged once its range is complete,
    which happens T - popcount(T) times; each merge stitches P pairs; each time step builds prod M leaves by
    ALS and P - prod M static nodes by STITCH."""
    P = int(np.prod([2 * sizes[m] - 1 for m in S])) if S else 1
    Mp = int(np.prod([sizes[m] for m in S])) if S else 1
    return (T - popcount(T)) * P + T * (P - Mp), T * Mp


def piece_bound(L):
    """canonical pieces of a length-L range on a segment tree: at most 2 per level of node size <= L"""
    return 2 * int(np.floor(np.log2(L))) + 2


# ------------------------------------------------------------------ data
def load_real(name):
    """memory-map, keep the time span, drop entities without data, crop the static modes; record what was kept"""
    X = np.load(osp.join(args.data_root, f'{name}.npy'), mmap_mode = 'r')
    T = X.shape[0] if args.T <= 0 else min(args.T, X.shape[0])
    want = args.static.get(name, [8] * (X.ndim - 1))
    keep, coverage = [], []
    sel = [slice(0, T)] + [slice(None)] * (X.ndim - 1)
    for mode in range(1, X.ndim):          # one mode at a time, measured inside what is already kept
        nz = np.zeros(X.shape[mode]); cnt = np.zeros(X.shape[mode])
        for c0 in range(0, T, 64):        # chunks over time: the files are up to 3.5 GB
            chunk = np.asarray(X[c0:min(T, c0 + 64)])
            for m2 in range(1, X.ndim):
                if m2 != mode and isinstance(sel[m2], np.ndarray):
                    chunk = np.take(chunk, sel[m2], axis = m2)
            axes = tuple(a for a in range(chunk.ndim) if a != mode)
            nz += (chunk != 0).sum(axis = axes); cnt += np.prod([chunk.shape[a] for a in axes])
        cov = nz / np.maximum(cnt, 1)
        good = np.where(cov >= args.min_coverage)[0]
        if len(good) < want[mode - 1]:    # not enough complete entities: take the best covered ones
            good = np.sort(np.argsort(-cov, kind = 'stable')[: want[mode - 1]])
        chosen = good[: want[mode - 1]]
        sel[mode] = chosen
        keep.append(chosen.tolist()); coverage.append(float(cov[chosen].mean()))
    Xs = np.asarray(X[:T, sel[1]])        # fancy index on the memory map: reads only the kept entities
    for mode in range(2, X.ndim):
        Xs = np.take(Xs, sel[mode], axis = mode)
    Xs = np.ascontiguousarray(Xs, dtype = np.float32)
    if args.standardize:
        axes = tuple(range(Xs.ndim - 1))
        Xs = (Xs - Xs.mean(axis = axes, keepdims = True)) / (Xs.std(axis = axes, keepdims = True) + 1e-8)
    meta = dict(full_shape = list(X.shape), used_shape = list(Xs.shape), T_used = T, kept_indices = keep,
                kept_mean_coverage = coverage, zero_fraction = float((Xs == 0).mean()))
    return torch.tensor(Xs, device = args.device), meta


def make_synthetic():
    """low-rank (core 3 x 2 x ... x 2) plus Gaussian noise, every static mode of size syn_M"""
    g = torch.Generator().manual_seed(args.seed)
    q, M, T = args.syn_modes, args.syn_M, args.syn_T
    core = torch.randn([3] + [2] * q, generator = g)
    U = [torch.randn(T, 3, generator = g)] + [torch.randn(M, 2, generator = g) for _ in range(q)]
    X = tensor_mats_mul(core, A_dim_list = [(U[i], i) for i in range(q + 1)])
    X = X + args.syn_noise * X.std() * torch.randn(X.shape, generator = g)
    return X.to(device = args.device, dtype = DTYPE), dict(used_shape = list(X.shape), T_used = T, synthetic = True)


# ------------------------------------------------------------------ design (fixed before any tree is built)
def ranks_for(name, sizes):
    if name in RANK_OVERRIDE:
        return RANK_OVERRIDE[name]
    return [args.time_rank] + [max(1, M // args.static_rank_div) for M in sizes[1:]]


def valid_lengths(M, r):
    """rank < length < size: shorter ranges are reconstructed exactly (nothing to test), the full range is no restriction"""
    return list(range(r + 1, M))


def clip_len(L, M, r):
    return int(min(max(round(L), r + 1), M - 1))


def time_lengths(T, rt):
    return sorted({int(min(max(round(f * T), rt + 1), T - 1)) for f in args.time_fracs})


def estimate_gb(T, sizes, ranks, S):
    """upper estimate of the tree's floats: P decompositions per temporal node (about 2T nodes): core, time
    factors (sum of node lengths over all levels = T (log2 T + 1)) and static factors (average node size)"""
    P = int(np.prod([2 * sizes[m] - 1 for m in S])) if S else 1
    core = int(np.prod(ranks))
    stat = sum(((np.log2(sizes[m]) + 1) / 2 if m in S else sizes[m]) * ranks[m] for m in range(1, len(sizes)))
    n = P * (2 * T * (core + stat) + T * (np.log2(max(T, 2)) + 1) * ranks[0])
    return n * 4 / 2 ** 30


def subsets(Q, synthetic):
    if synthetic:                       # equal, exchangeable modes: prefixes are enough
        return [tuple(Q[:k]) for k in range(min(len(Q), args.syn_max_k) + 1)]
    return [S for k in range(len(Q) + 1) for S in itertools.combinations(Q, k)]


def design(name, X, ranks, Q, synthetic):
    """all queries of a data set. Paired: every repetition j has one time window per time length and one
    relative position u_j[m] in [0, 1) per static mode, reused by every setting."""
    T, sizes = X.size(0), list(X.shape)
    rng = np.random.default_rng([args.seed, zlib.crc32(name.encode())])
    tls = time_lengths(T, ranks[0])
    windows = {}
    for Lt in tls:                       # distinct starts, uniform over all valid positions
        n_pos = T - Lt + 1
        starts = rng.choice(n_pos, size = args.n_reps, replace = n_pos < args.n_reps)
        windows[Lt] = [(int(s), int(s) + Lt) for s in starts]
    u = rng.random((args.n_reps, len(sizes)))

    def place(m, L, j):
        lo = int(np.floor(u[j, m] * (sizes[m] - L + 1)))
        return (lo, lo + L)

    qs = []
    if ('k' in EXPS and not synthetic) or (synthetic and 'synth' in EXPS):
        for S in subsets(Q, synthetic):
            k = len(S)
            for Lt in tls:
                for j in range(args.n_reps):
                    t0, t1 = windows[Lt][j]
                    qs.append(dict(exp = 'k', family = 'control_full', S = list(S), k = k, rho = 1.0, Lt = Lt, rep = j,
                                   t0 = t0, t1 = t1, ranges = {m: (0, sizes[m]) for m in S}))
                    if k == 0:
                        continue
                    for rho in args.rho:
                        rngs = {m: place(m, clip_len(sizes[m] * rho ** (1 / k), sizes[m], ranks[m]), j) for m in S}
                        qs.append(dict(exp = 'k', family = 'matched', S = list(S), k = k, rho = rho, Lt = Lt, rep = j,
                                       t0 = t0, t1 = t1, ranges = rngs))
    if 'length' in EXPS and not synthetic and Q:
        Lt_al = 2 ** int(np.floor(np.log2(T)))  # one complete, aligned temporal node: 1 hit, no time slicing
        for m in Q:
            for L in valid_lengths(sizes[m], ranks[m]):
                n_pos = sizes[m] - L + 1
                starts = rng.choice(n_pos, size = min(args.n_reps, n_pos), replace = False)
                for j, lo in enumerate(sorted(int(s) for s in starts)):
                    qs.append(dict(exp = 'length', family = f'mode{m}', S = list(Q), k = len(Q), rho = None, Lt = Lt_al,
                                   rep = j, t0 = 0, t1 = Lt_al, ranges = {m: (lo, lo + L)}))
        if len(Q) >= 2:
            m1, m2 = Q[:2]
            for f1, f2 in itertools.product(args.len_grid_fracs, args.len_grid_fracs):
                L1, L2 = clip_len(f1 * sizes[m1], sizes[m1], ranks[m1]), clip_len(f2 * sizes[m2], sizes[m2], ranks[m2])
                for j in range(args.n_reps):
                    qs.append(dict(exp = 'length', family = 'grid', S = list(Q), k = len(Q), rho = None, Lt = Lt_al, rep = j,
                                   t0 = 0, t1 = Lt_al, ranges = {m1: place(m1, L1, j), m2: place(m2, L2, j)}))
    if 'stream' in EXPS and not synthetic:
        cps = []
        for f in args.checkpoints:
            c = int(round(f * T))
            if c & (c - 1) == 0:                 # avoid powers of two: they leave no incomplete node
                c -= 1
            cps.append(c)
        rho = args.rho[0]
        for S in subsets(Q, synthetic):
            for c in sorted(set(cps)):
                for Lt in tls:
                    if Lt >= c:
                        continue
                    for j in range(args.n_reps):    # window ends at the newest slice
                        rngs = {m: place(m, clip_len(sizes[m] * rho ** (1 / len(S)), sizes[m], ranks[m]), j) for m in S} if S else {}
                        qs.append(dict(exp = 'stream', family = f'T={c}', S = list(S), k = len(S), rho = rho if S else 1.0,
                                       Lt = Lt, rep = j, t0 = c - Lt, t1 = c, ranges = rngs, checkpoint = c))
    for q in qs:                                 # sanity: every range must satisfy the design
        for m, (lo, hi) in q['ranges'].items():
            assert 0 <= lo < hi <= sizes[m], (q, sizes)
            if q['family'] != 'control_full':
                assert ranks[m] < hi - lo < sizes[m], (q, ranks, sizes)
        assert 0 <= q['t0'] < q['t1'] <= T
    return qs, dict(time_lengths = tls, length_levels = {m: valid_lengths(sizes[m], ranks[m]) for m in Q})


# ------------------------------------------------------------------ evaluation
ALS_CACHE = {}


def full_ranges(sizes, ranges):
    return tuple(tuple(ranges.get(m, (0, sizes[m]))) for m in range(1, len(sizes)))


def err64(Xq, tucker):
    Xh = tensor_mats_mul(tucker.G.double(), A_dim_list = [(U.double(), p) for p, U in enumerate(tucker.U)])
    return float(torch.linalg.norm((Xq.double() - Xh).flatten()))


def evaluate(name, tree, X, q, ranks):
    sizes = list(X.shape)
    box = full_ranges(sizes, q['ranges'])
    Xq = X[(slice(q['t0'], q['t1']),) + tuple(slice(lo, hi) for lo, hi in box)]
    norm = float(torch.linalg.norm(Xq.double().flatten()))
    row = dict(dataset = name, exp = q['exp'], family = q['family'], S = '-'.join(map(str, q['S'])) or 'none', k = q['k'],
               rho = q['rho'], Lt = q['Lt'], rep = q['rep'], t0 = q['t0'], t1 = q['t1'],
               box = ';'.join(f'{lo}:{hi}' for lo, hi in box), box_entries = int(Xq.numel()),
               checkpoint = q.get('checkpoint', ''))
    for m in range(1, len(sizes)):
        row[f'L{m}'] = box[m - 1][1] - box[m - 1][0]
    if norm == 0.:
        row.update(skipped = 'zero_box')
        return row
    times = []
    for _ in range(args.time_reps):
        set_seed(args.seed + 1)
        reset_op_count()
        tic = now()
        tucker, _, hits = tree.query_tucker(q['t0'], q['t1'], args.prune, ranges = q['ranges'])
        times.append(now() - tic)
    st = tree.last_stats
    for m in range(1, len(sizes)):
        row[f'pieces{m}'] = st['pieces'].get(m, '')
    row.update(hits = hits, partial_hits = st['partial_hits'], leaf_pieces = st['leaf_pieces'],
               stitch = OP_COUNT['stitch'], partial = OP_COUNT['partial'], query_ms = 1e3 * float(np.median(times)),
               err_tree = err64(Xq, tucker) / norm)
    key = (name, q['t0'], q['t1'], box)
    if key not in ALS_CACHE:                     # same box in another setting: same reference
        set_seed(args.seed + 1)
        tic = now()
        tk, _ = tucker_als(Xq, ranks, args.als_tol, args.als_maxiters)
        ALS_CACHE[key] = (err64(Xq, tk) / norm, now() - tic)
    row['err_als'], als_s = ALS_CACHE[key]
    row['als_ms'] = 1e3 * als_s
    row['trivial'] = int(row['err_als'] < args.trivial)
    row['eq29'] = '' if row['trivial'] else row['err_tree'] / row['err_als'] - 1
    eff = [min(r, hi - lo) for r, (lo, hi) in zip(ranks, ((q['t0'], q['t1']),) + box)]
    row['tucker_params_over_entries'] = (np.prod(eff) + sum(e * (hi - lo) for e, (lo, hi) in zip(eff, ((q['t0'], q['t1']),) + box))) / Xq.numel()
    return row


def run_setting(name, X, S, ranks, queries, rows, builds):
    """build the tree on modes S (one slice at a time), answering the stream queries at their checkpoints"""
    T, sizes = X.size(0), list(X.shape)
    S = list(S)
    mine = [q for q in queries if q['S'] == S]
    stream = {}
    for q in mine:
        if q['exp'] == 'stream':
            stream.setdefault(q['checkpoint'], []).append(q)
    set_seed(args.seed)
    reset_op_count()
    tree = TucketTree(ranks, args.tol, args.maxiters, alloc = T, query_dims = S)
    lat = []
    tic0 = now()
    for t in range(T):
        tic = now()
        tree.append(X[t])
        lat.append(now() - tic)
        if t + 1 in stream:
            ops = dict(OP_COUNT)
            for q in stream[t + 1]:
                rows.append(evaluate(name, tree, X, q, ranks))
            OP_COUNT.update(ops)                  # do not count query work as build work
    build_s = sum(lat)
    ops = dict(OP_COUNT)
    f_stitch, f_als = formula_build(T, sizes, S)
    n_tk, n_fl, n_use = tree.storage()
    builds.append(dict(dataset = name, S = '-'.join(map(str, S)) or 'none', k = len(S), T = T,
                       P = int(np.prod([2 * sizes[m] - 1 for m in S])) if S else 1, build_s = build_s,
                       append_ms_mean = 1e3 * float(np.mean(lat)), append_ms_p99 = 1e3 * float(np.percentile(lat, 99)),
                       append_ms_max = 1e3 * float(np.max(lat)), build_stitch = ops['stitch'], formula_stitch = f_stitch,
                       build_als = ops['als'], formula_als = f_als,
                       formula_ok = int(ops['stitch'] == f_stitch and ops['als'] == f_als),
                       n_tuckers = n_tk, n_floats = n_fl, n_floats_no_padding = n_use, raw_floats = X.numel()))
    log(f'  S={S or "none"}: build {build_s:.1f}s, STITCH {ops["stitch"]} (formula {f_stitch}), ALS {ops["als"]} '
        f'(formula {f_als}), storage {n_fl / X.numel():.1f}x raw ({n_use / X.numel():.1f}x without padding)')
    first = True
    for q in mine:
        if q['exp'] == 'stream':
            continue
        if first:                                 # untimed warm-up
            tree.query_tucker(q['t0'], q['t1'], args.prune, ranges = q['ranges'])
            first = False
        rows.append(evaluate(name, tree, X, q, ranks))
    del tree


# ------------------------------------------------------------------ run
config = dict(vars(args)); config['static'] = args.static; config['started'] = stamp
config['torch'] = torch.__version__; config['datasets_used'] = {}
rows, builds = [], []
jobs = [n for n in args.datasets if 'k' in EXPS or 'length' in EXPS or 'stream' in EXPS]
if 'synth' in EXPS:
    jobs.append('Synthetic')
for name in jobs:
    synthetic = name == 'Synthetic'
    X, meta = make_synthetic() if synthetic else load_real(name)
    sizes = list(X.shape)
    ranks = ranks_for(name, sizes) if not synthetic else [3] + [2] * args.syn_modes
    Q = [m for m in range(1, len(sizes)) if sizes[m] >= ranks[m] + 2]   # modes where rank < length < size is possible
    gb = estimate_gb(X.size(0), sizes, ranks, Q)
    if gb > args.max_gb:                         # keep the build feasible; the reduction is logged and saved
        T_new = int(X.size(0) * args.max_gb / gb)
        log(f'== {name}: estimated {gb:.1f} GB > --max_gb {args.max_gb}; using the first {T_new} time steps')
        X = X[:T_new].contiguous(); meta['T_used'] = T_new; meta['T_reduced_for_memory'] = True
    queries, levels = design(name, X, ranks, Q, synthetic)
    meta.update(ranks = ranks, queryable_modes = Q, est_gb = estimate_gb(X.size(0), sizes, ranks, Q), **levels)
    config['datasets_used'][name] = meta
    with open(osp.join(OUT, f'queries_{name}.json'), 'w') as f:
        json.dump(queries, f)
    with open(osp.join(OUT, 'config.json'), 'w') as f:
        json.dump(config, f, indent = 1, default = str)
    n_stitch = sum(formula_build(X.size(0), sizes, S)[0] for S in subsets(Q, synthetic))
    meta['est_build_stitch_total'] = n_stitch
    log(f'== {name}: shape {tuple(X.shape)}, ranks {ranks}, queryable modes {Q}, {len(queries)} queries, '
        f'est. tree memory {meta["est_gb"]:.2f} GB, {n_stitch / 1e6:.2f}M build STITCH calls in total '
        f'(about {n_stitch * 1.5e-3 / 3600:.1f} h at 1.5 ms each; use --T / --static to shrink)')
    for S in subsets(Q, synthetic):
        if not any(q['S'] == list(S) for q in queries):
            continue
        run_setting(name, X, S, ranks, queries, rows, builds)
    ALS_CACHE.clear()

    def save(fname, data):
        keys = []
        for r in data:
            keys += [k for k in r if k not in keys]
        with open(osp.join(OUT, fname), 'w', newline = '') as f:
            w = csv.DictWriter(f, fieldnames = keys)
            w.writeheader()
            w.writerows(data)
    save('queries.csv', rows)
    save('builds.csv', builds)

# ------------------------------------------------------------------ plots and summary
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, GRAY = '#2a78d6', '#eb6834', '#2e9e6b', '#6b6a64'


def num(r, k):
    v = r.get(k, '')
    return float(v) if v not in ('', None) else np.nan


def med_iqr(v):
    v = np.array([x for x in v if np.isfinite(x)])
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


def clean(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha = 0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


names = list(dict.fromkeys(r['dataset'] for r in rows))
QTIME = 'query time: O(log T + hits * prod_i pieces_i * stitch cost), pieces_i <= 2 floor(log2 L_i) + 2'

# 1. Eq. 29 vs k (median and IQR) per family; second row: the ALS reference error (should be flat for "matched")
krows = [r for r in rows if r['exp'] == 'k' and 'skipped' not in r]
if krows:
    fig, axes = plt.subplots(2, len(names), figsize = (4.2 * len(names), 6.4), squeeze = False)
    for j, name in enumerate(names):
        R = [r for r in krows if r['dataset'] == name]
        ks = sorted({r['k'] for r in R})
        series = [('control_full', 1.0, GRAY, 'same box for every k (static modes in full)')] + \
                 [('matched', rho, c, f'static volume fixed: rho={rho}') for rho, c in zip(args.rho, (BLUE, ORANGE, GREEN))]
        for fam, rho, c, lab in series:
            xs, md, lo, hi, al = [], [], [], [], []
            for k in ks:
                v = [num(r, 'eq29') for r in R if r['k'] == k and r['family'] == fam and (fam == 'control_full' or r['rho'] == rho)]
                a = [num(r, 'err_als') for r in R if r['k'] == k and r['family'] == fam and (fam == 'control_full' or r['rho'] == rho)]
                if not v:
                    continue
                m, l, h = med_iqr(v)
                xs.append(k); md.append(100 * m); lo.append(100 * (m - l)); hi.append(100 * (h - m)); al.append(med_iqr(a)[0])
            if xs:
                axes[0][j].errorbar(xs, md, yerr = [lo, hi], color = c, marker = 'o', capsize = 3, label = lab)
                axes[1][j].plot(xs, al, color = c, marker = 's', label = lab)
        axes[0][j].axhline(0, color = GRAY, lw = 1, ls = ':')
        axes[0][j].set_title(name)
        clean(axes[0][j], '# query modes k (tree on exactly those modes)', 'Eq. 29: err_tree / err_ALS - 1 (%)')
        clean(axes[1][j], '# query modes k', 'Tucker-ALS reference error (median)')
    axes[0][0].legend(fontsize = 7); axes[1][0].legend(fontsize = 7)
    fig.suptitle('Accuracy vs number of query modes (median, IQR; trivial and all-zero boxes excluded)')
    fig.tight_layout(); fig.savefig(osp.join(OUT, 'fig_k_error.png'), dpi = 150); plt.close(fig)

# 2. cost vs k: build STITCH measured vs general formula, storage vs P, query time vs prod of pieces
if builds:
    bn = list(dict.fromkeys(b['dataset'] for b in builds))
    fig, axes = plt.subplots(3, len(bn), figsize = (4.2 * len(bn), 9.3), squeeze = False)
    for j, name in enumerate(bn):
        B = [b for b in builds if b['dataset'] == name]
        P = [b['P'] for b in B]
        axes[0][j].scatter(P, [b['build_stitch'] for b in B], color = BLUE, label = 'measured', zorder = 3)
        axes[0][j].scatter(P, [b['formula_stitch'] for b in B], color = ORANGE, marker = 'x', s = 60, label = 'formula', zorder = 4)
        axes[0][j].set_xscale('log'); axes[0][j].set_yscale('log'); axes[0][j].set_title(name)
        clean(axes[0][j], 'P = prod_{i in S} (2 M_i - 1)', 'build STITCH calls')
        axes[1][j].scatter(P, [b['n_floats'] / b['raw_floats'] for b in B], color = BLUE, label = 'stored')
        axes[1][j].scatter(P, [b['n_floats_no_padding'] / b['raw_floats'] for b in B], color = GREEN, label = 'without zero padding')
        axes[1][j].set_xscale('log'); axes[1][j].set_yscale('log')
        clean(axes[1][j], 'P', 'storage / raw data')
        R = [r for r in rows if r['dataset'] == name and r['exp'] == 'k' and 'skipped' not in r]
        axes[2][j].scatter([num(r, 'leaf_pieces') for r in R], [num(r, 'query_ms') for r in R], s = 8, color = BLUE, alpha = 0.6)
        axes[2][j].set_xscale('log'); axes[2][j].set_yscale('log')
        clean(axes[2][j], 'leaf pieces visited (= hits x prod_i pieces_i)', 'query time (ms)')
    axes[0][0].legend(fontsize = 7); axes[1][0].legend(fontsize = 7)
    fig.suptitle('Cost.  Build STITCH = (T - popcount T) P + T (P - prod M_i);  space O(P T (r^p + r D + r log T));\n' + QTIME,
                 fontsize = 9)
    fig.tight_layout(); fig.savefig(osp.join(OUT, 'fig_k_cost.png'), dpi = 150); plt.close(fig)

# 3. length experiment: pieces vs L with the bound, and query time vs pieces
lrows = [r for r in rows if r['exp'] == 'length' and 'skipped' not in r]
if lrows:
    ln = list(dict.fromkeys(r['dataset'] for r in lrows))
    fig, axes = plt.subplots(2, len(ln), figsize = (4.2 * len(ln), 6.4), squeeze = False)
    for j, name in enumerate(ln):
        R = [r for r in lrows if r['dataset'] == name and r['family'].startswith('mode')]
        for fam, c in zip(sorted({r['family'] for r in R}), (BLUE, ORANGE, GREEN)):
            m = int(fam[4:])
            F = [r for r in R if r['family'] == fam]
            Ls = sorted({r[f'L{m}'] for r in F})
            mx = [max(num(r, f'pieces{m}') for r in F if r[f'L{m}'] == L) for L in Ls]
            mn = [np.mean([num(r, f'pieces{m}') for r in F if r[f'L{m}'] == L]) for L in Ls]
            axes[0][j].plot(Ls, mx, color = c, marker = 'o', label = f'mode {m}: max over positions')
            axes[0][j].plot(Ls, mn, color = c, ls = '--', label = f'mode {m}: mean')
            axes[1][j].scatter([num(r, f'pieces{m}') for r in F], [num(r, 'query_ms') for r in F], s = 10, color = c, alpha = 0.6,
                               label = f'mode {m}')
        Lall = np.arange(2, max(r[f'L{int(f[4:])}'] for r in R for f in [r['family']]) + 1)
        axes[0][j].plot(Lall, [piece_bound(L) for L in Lall], color = GRAY, ls = ':', label = 'bound 2 floor(log2 L) + 2')
        axes[0][j].set_title(name)
        clean(axes[0][j], 'range length L', 'canonical pieces')
        clean(axes[1][j], 'canonical pieces on the varied mode', 'query time (ms)')
        axes[0][j].legend(fontsize = 6); axes[1][j].legend(fontsize = 6)
    fig.suptitle('Query cost vs range length (aligned time node: 1 hit, no time slicing; other modes in full)\n' + QTIME, fontsize = 9)
    fig.tight_layout(); fig.savefig(osp.join(OUT, 'fig_length.png'), dpi = 150); plt.close(fig)

# 4. stream: Eq. 29 at checkpoints during the build
srows = [r for r in rows if r['exp'] == 'stream' and 'skipped' not in r]
if srows:
    sn = list(dict.fromkeys(r['dataset'] for r in srows))
    fig, axes = plt.subplots(1, len(sn), figsize = (4.2 * len(sn), 3.4), squeeze = False)
    for j, name in enumerate(sn):
        R = [r for r in srows if r['dataset'] == name]
        for k, c in zip(sorted({r['k'] for r in R}), (GRAY, BLUE, ORANGE, GREEN)):
            cps = sorted({int(r['checkpoint']) for r in R if r['k'] == k})
            md = [100 * med_iqr([num(r, 'eq29') for r in R if r['k'] == k and int(r['checkpoint']) == cp])[0] for cp in cps]
            axes[0][j].plot(cps, md, color = c, marker = 'o', label = f'k={k}')
        axes[0][j].set_title(name); clean(axes[0][j], 'checkpoint (slices appended so far)', 'Eq. 29 (%), median')
        axes[0][j].legend(fontsize = 7)
    fig.suptitle('Queries during the build, on windows ending at the newest slice (incomplete right spine)', fontsize = 9)
    fig.tight_layout(); fig.savefig(osp.join(OUT, 'fig_stream.png'), dpi = 150); plt.close(fig)

# summary
with open(osp.join(OUT, 'summary.txt'), 'w') as f:
    for name in names:
        R = [r for r in rows if r['dataset'] == name]
        B = [b for b in builds if b['dataset'] == name]
        ev = [r for r in R if 'skipped' not in r]
        triv = sum(int(r['trivial']) for r in ev)
        e29 = [num(r, 'eq29') for r in ev]
        lr = [r for r in ev if r['exp'] == 'length' and r['family'].startswith('mode')]
        viol = sum(num(r, f"pieces{r['family'][4:]}") > piece_bound(r[f"L{r['family'][4:]}"]) for r in lr)
        f.write(f'{name}: {len(R)} queries, {len(R) - len(ev)} all-zero boxes skipped, {triv} trivial (ALS error < {args.trivial}); '
                f'Eq. 29 median {100 * med_iqr(e29)[0]:.2f}%; build formula ok for {sum(b["formula_ok"] for b in B)}/{len(B)} '
                f'settings; piece bound violated by {viol}/{len(lr)} length queries\n')
log(open(osp.join(OUT, 'summary.txt')).read())
log(f'saved results to {OUT}/')
