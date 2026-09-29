from inc.utils import *
from inc.tucker import *

# ---------------------------------------------------------------------------
# Multidimensional extension: range queries on the non-temporal modes in `query_dims`
#
# Every non-temporal mode listed in `query_dims` supports range queries and gets
# its own complete (static: its size is known up front) segment tree, nested
# inside the previous one: every node of the tree over query_dims[0] carries in
# `.inner` a structure of the same kind over query_dims[1:], down to a flat
# Tucker decomposition (StaticLeaf).
# Non-temporal modes NOT in query_dims get no tree and cannot be restricted by a
# query: they are always taken in full, so they add nothing to the cost.
#
# With query_dims = [] every payload is a single StaticLeaf and everything
# below reduces to the original TUCKET code path.
# ---------------------------------------------------------------------------

class StaticLeaf:
    """base case of the nested structure: one flat Tucker decomposition"""
    __slots__ = ('tucker', 'norm2')
    def __init__(self, tucker, norm2):
        self.tucker = tucker
        self.norm2 = norm2

class StaticAxisNode:
    """node of a complete static segment tree over mode `axis`; `.inner` is
    the nested structure (over the remaining query_dims) for [i0, i1) on `axis`"""
    __slots__ = ('axis', 'i0', 'i1', 'l', 'r', 'inner', 'norm2')
    def __init__(self, axis, i0, i1, l, r, inner, norm2):
        self.axis = axis
        self.i0 = i0
        self.i1 = i1
        self.l = l
        self.r = r
        self.inner = inner
        self.norm2 = norm2

def build_nested(X, query_dims, ranks, tol, maxiters):
    """build the nested structure for a tensor X (full range on every tree dim)"""
    if not query_dims:
        norm2 = tensor_norm2(X)
        tucker, fit = tucker_als(X, ranks, tol, maxiters, norm2 = norm2, verbose = False)
        return StaticLeaf(tucker = tucker, norm2 = norm2)
    return _build_axis_tree(X, query_dims[0], 0, X.size(dim = query_dims[0]), query_dims[1 :], ranks, tol, maxiters)

def _build_axis_tree(X, axis, i0, i1, rest_dims, ranks, tol, maxiters):
    """bottom-up construction of the static tree over `axis` on [i0, i1)"""
    if i1 - i0 == 1:
        inner = build_nested(X.narrow(axis, i0, 1), rest_dims, ranks, tol, maxiters)
        return StaticAxisNode(axis = axis, i0 = i0, i1 = i1, l = None, r = None, inner = inner, norm2 = inner.norm2)
    im = (i0 + i1) >> 1
    l = _build_axis_tree(X, axis, i0, im, rest_dims, ranks, tol, maxiters)
    r = _build_axis_tree(X, axis, im, i1, rest_dims, ranks, tol, maxiters)
    inner = stitch_nested(l.inner, r.inner, axis = axis, ranks = ranks, tol = tol, maxiters = maxiters)
    return StaticAxisNode(axis = axis, i0 = i0, i1 = i1, l = l, r = r, inner = inner, norm2 = l.norm2 + r.norm2)

def stitch_nested(a, b, axis, ranks, tol, maxiters):
    """stitch two same-shaped nested structures along mode `axis`: walk them in
    lockstep (they are node-for-node isomorphic since their shape only depends
    on the fixed sizes of query_dims) and STITCH every pair of flat leaves"""
    norm2 = a.norm2 + b.norm2
    if isinstance(a, StaticLeaf):
        tucker, fit = tucker_stitch([a.tucker, b.tucker], ranks, tol, maxiters, norm2 = norm2, verbose = False, axis = axis)
        return StaticLeaf(tucker = tucker, norm2 = norm2)
    inner = stitch_nested(a.inner, b.inner, axis, ranks, tol, maxiters)
    if a.l is None:
        l = r = None
    else:
        l = stitch_nested(a.l, b.l, axis, ranks, tol, maxiters)
        r = stitch_nested(a.r, b.r, axis, ranks, tol, maxiters)
    return StaticAxisNode(axis = a.axis, i0 = a.i0, i1 = a.i1, l = l, r = r, inner = inner, norm2 = norm2)

def _static_decompose(node, a, b):
    """canonical decomposition of [a, b) on a complete static tree; returns the `.inner`s of the covering nodes"""
    if a <= node.i0 and node.i1 <= b:
        return [node.inner]
    im = (node.i0 + node.i1) >> 1
    out = []
    if a < im:
        out += _static_decompose(node.l, a, min(b, im))
    if b > im:
        out += _static_decompose(node.r, max(a, im), b)
    return out

def query_nested(nested, query_dims, ranges, ranks, tol, maxiters, time_range = None, qr = True):
    """CombineHitSet on one temporal hit. Order matters for cost:
    1. static decomposition first: resolve query_dims one mode at a time (canonical
       decomposition of the query range, recurse into each piece, STITCH the pieces
       along that mode) -- only O(log L_i) pieces per mode are ever visited;
    2. then, only at the leaves actually reached, cut the temporal mode to
       `time_range` if the hit is only partially covered in time.
    `ranges` maps mode -> (lo, hi); a mode absent from `ranges` is taken in full."""
    if not query_dims:
        tucker = nested.tucker
        if time_range is not None:
            tucker = tucker_partial(tucker, t0 = time_range[0], t1 = time_range[1], qr = qr)
        return tucker, nested.norm2
    axis = query_dims[0]
    lo, hi = ranges.get(axis, (nested.i0, nested.i1))
    tuckers, norm2 = [], 0.
    for piece in _static_decompose(nested, lo, hi):
        tucker, n2 = query_nested(piece, query_dims[1 :], ranges, ranks, tol, maxiters, time_range, qr)
        tuckers.append(tucker)
        norm2 += n2
    if len(tuckers) == 1:
        return tuckers[0], norm2
    tucker, fit = tucker_stitch(tuckers, ranks, tol, maxiters, norm2 = norm2, verbose = False, axis = axis)
    return tucker, norm2

# ---------------------------------------------------------------------------
# TUCKET's stream segment tree on the temporal mode (payload generalized from
# a flat Tucker decomposition to a nested structure over query_dims)
# ---------------------------------------------------------------------------

class TucketNode:
    """node of TUCKET"""
    def __init__(self, t0, t1, l = None, r = None, built = None, norm2 = 0., nested = None):
        self.t0 = t0
        self.t1 = t1
        self.l = l
        self.r = r
        self.built = built
        self.norm2 = norm2
        self.nested = nested
    @property
    def tm(self):
        """middle time"""
        return (self.t0 + self.t1) >> 1
    @property
    def tlen(self):
        """time range length"""
        return self.t1 - self.t0
    @property
    def tucker(self):
        """the Tucker decomposition of the whole node (all non-temporal modes in full)"""
        nested = self.nested
        while not isinstance(nested, StaticLeaf):
            nested = nested.inner
        return nested.tucker
    def build(self, X, query_dims, ranks, tol, maxiters):
        """preprocess the (nested) Tucker decompositions of the node"""
        self.norm2 = tensor_norm2(X)
        self.nested = build_nested(X, query_dims, ranks, tol, maxiters) # not using mat_svd_eigh due to its numerical instability
        self.built = True
    def stitch(self, query_dims, ranks, tol, maxiters):
        """stitch the (nested) Tucker decompositions of children nodes along the temporal mode"""
        self.norm2 = self.l.norm2 + self.r.norm2
        self.nested = stitch_nested(self.l.nested, self.r.nested, axis = 0, ranks = ranks, tol = tol, maxiters = maxiters)
        self.built = True
    def partial_tucker(self, t0, t1, qr = True):
        """approximate a subtensor Tucker decomposition"""
        return tucker_partial(self.tucker, t0 = max(t0, self.t0) - self.t0, t1 = min(t1, self.t1) - self.t0, qr = qr)

class TucketTree: # time starts from 0
    """stream segment tree of TUCKET; `query_dims` lists the non-temporal modes that
    support range queries (each gets a nested segment tree); the other non-temporal
    modes get no tree and are always taken in full. [] is the original TUCKET"""
    def __init__(self, ranks, tol, maxiters, alloc = 1, query_dims = ()):
        self.cfg = Dict(ranks = deepcopy(ranks), tol = tol, maxiters = maxiters)
        self.n_dims = len(ranks)
        self.query_dims = list(query_dims)
        if any(not 0 < p < self.n_dims for p in self.query_dims) or len(set(self.query_dims)) != len(self.query_dims):
            raise ValueError(f'query_dims must be distinct non-temporal modes in 1..{self.n_dims - 1}, got {self.query_dims}')
        self.tlen = 0
        self.root = None
        self.n_nodes = 0
        self.alloc = alloc
        self.nodes = [None for _ in range(2 ** (math.ceil(math.log2(alloc)) + 1) - 1)]
        ##self.hits_logs = []
    def _new_node(self, t0, t1, **kwargs):
        """create a new node"""
        node_id = self.n_nodes
        self.n_nodes += 1
        if node_id >= len(self.nodes):
            self.nodes.append(None)
        self.nodes[node_id] = TucketNode(t0 = t0, t1 = t1, **kwargs)
        return self.nodes[node_id]
    def _insert(self, node, t, Xt):
        """insert a leaf node"""
        if t == node.t0 and t + 1 == node.t1: # leaf node
            node.build(X = Xt, query_dims = self.query_dims, **self.cfg)
        elif t < node.tm: # go to left child
            if node.l is None:
                node.l = self._new_node(node.t0, node.tm)
            self._insert(node.l, t, Xt)
        else: # go to right child
            if node.r is None:
                node.r = self._new_node(node.tm, node.t1)
            self._insert(node.r, t, Xt)
            if t + 1 == node.t1:
                node.stitch(query_dims = self.query_dims, **self.cfg)
    def append(self, Xt):
        """append a tensor slice"""
        t = self.tlen
        self.tlen += 1
        if self.root is None:
            self.root = self._new_node(t0 = 0, t1 = 1)
        elif t >= self.root.t1:
            old_root = self.root
            self.root = self._new_node(t0 = 0, t1 = old_root.t1 << 1, l = old_root)
        self._insert(self.root, t, Xt[None])
    def _recall(self, node, t0, t1, prune, hits: list):
        """find a pruned hit set"""
        if node.built and (t1 - t0) >= node.tlen * prune:
            hits.append(node)
        elif t1 <= node.tm:
            self._recall(node.l, t0, t1, prune, hits)
        elif t0 >= node.tm:
            self._recall(node.r, t0, t1, prune, hits)
        else:
            self._recall(node.l, t0, node.tm, prune, hits)
            self._recall(node.r, node.tm, t1, prune, hits)
    def _query_norm2(self, node, t0, t1):
        """find the squared norm of the subtensor [t0, t1)"""
        if t0 == node.t0 and t1 == node.t1:
            return node.norm2
        elif t1 <= node.tm:
            return self._query_norm2(node.l, t0, t1)
        elif t0 >= node.tm:
            return self._query_norm2(node.r, t0, t1)
        else:
            return self._query_norm2(node.l, t0, node.tm) + self._query_norm2(node.r, node.tm, t1)
    def query_tucker(self, t0, t1, prune, ranges = None): # [t0, t1)
        """answer a range query by stitching the hit set; `ranges` optionally maps
        modes in query_dims to (lo, hi) ranges (absent modes are taken in full)"""
        ranges = dict(ranges or {})
        bad = [p for p in ranges if p not in self.query_dims]
        if bad:
            raise ValueError(f'modes {bad} have no tree (query_dims = {self.query_dims}); they can only be taken in full')
        norm2 = self._query_norm2(self.root, t0, t1)
        hits = []
        self._recall(self.root, t0 = t0, t1 = t1, prune = prune, hits = hits)
        tuckers = []
        for node in hits:
            if t0 <= node.t0 and t1 >= node.t1: # fully covered in time
                time_range = None
            else: # partially covered: cut time only at the leaves the static decomposition uses
                time_range = (max(t0, node.t0) - node.t0, min(t1, node.t1) - node.t0)
            tucker, _ = query_nested(node.nested, self.query_dims, ranges, **self.cfg,
                                     time_range = time_range, qr = len(hits) < 2)
            tuckers.append(tucker)
        tucker, fit = tucker_stitch(tuckers = tuckers, norm2 = norm2, **self.cfg, mat_svd_fn = mat_svd_eigh)
        return tucker, fit, len(hits)
    def storage(self):
        """(number of stored Tucker decompositions, number of stored floats) -- for measuring space"""
        n_tuckers, n_floats, stack = 0, 0, [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            stack += [c for c in (node.l, node.r) if c is not None]
            if not node.built:
                continue
            inner = [node.nested]
            while inner:
                x = inner.pop()
                if isinstance(x, StaticLeaf):
                    n_tuckers += 1
                    n_floats += x.tucker.G.numel() + sum(Up.numel() for Up in x.tucker.U)
                else:
                    inner += [x.inner] + [c for c in (x.l, x.r) if c is not None]
        return n_tuckers, n_floats

def random_ranges(rng, sizes, query_dims):
    """a random range (length >= 2) on every mode in query_dims"""
    ranges = dict()
    for axis in query_dims:
        lo = int(rng.integers(0, sizes[axis] - 1))
        hi = lo + int(rng.integers(2, sizes[axis] - lo + 1))
        ranges[axis] = (lo, hi)
    return ranges

if __name__ == '__main__':
    from inc.args import *

    parser = ArgParser(prog = TucketTree.__name__)
    parser.add_argument('--prune', type = float, default = 0.7, help = 'pruning threshold of TUCKET')
    parser.add_argument('--query_dims', type = lambda s: parse_ints(s) if s else [], default = [], help = 'non-temporal modes that support range queries, e.g. 1,2 (a tree is built on each; empty = original TUCKET); every query gets a random range on each of them')
    args = parser.parse_args()

    from inc.data import *

    X = load_data(root = args.data_root, name = args.dataset, device = args.device)
    tlen = X.size(dim = 0)
    sizes = list(X.shape)

    tree = TucketTree(ranks = args.ranks, tol = args.tol, maxiters = args.maxiters, alloc = tlen, query_dims = args.query_dims)
    res = Dict(dur = [], mem = [])
    tbar = trange(tlen)
    for t in tbar:
        mem0 = torch.cuda.memory_allocated(0)
        tic = time.time()
        tree.append(X[t])
        toc = time.time()
        mem1 = torch.cuda.memory_allocated(0)
        res.dur.append(toc - tic)
        res.mem.append((mem1 - mem0) / 1073741824)
        tbar.set_description(f'cummem={mem1 / 1073741824:.2f}GB')
    res = dict(res)
    with open(f'{args.save_name}~eval~append.pkl', 'wb') as fo:
        pkl.dump(res, fo)

    queries = load_queries(root = args.queries_root, name = args.dataset)
    rng = np.random.default_rng(args.seed)
    res = Dict(qlen = [], ranges = [], orig = [], err = [], dur = [], hits = [])
    with torch.no_grad():
        for qlen, ts in queries.items():
            for t0, t1 in tqdm(ts, desc = f'qlen={qlen}'):
                ranges = random_ranges(rng, sizes, args.query_dims)
                tic = time.time()
                tucker, _, hits = tree.query_tucker(t0, t1, prune = args.prune, ranges = ranges)
                toc = time.time()
                res.qlen.append(qlen)
                res.ranges.append(ranges)
                Xq = X[tuple([slice(t0, t1)] + [slice(*ranges[p]) if p in ranges else slice(None) for p in range(1, len(sizes))])]
                res.orig.append(tensor_norm(Xq).item())
                res.err.append(tensor_norm(Xq - tensor_mats_mul(tucker.G, A_dim_list = [(Up, p) for p, Up in enumerate(tucker.U)])).item())
                res.dur.append(toc - tic)
                res.hits.append(hits)
            print(f'[qlen={qlen}] avg_err={(np.array(res.err[-len(ts) :]) / np.array(res.orig[-len(ts) :])).mean():.4f}', flush = True)
    res = dict(res)
    with open(f'{args.save_name}~eval~query.pkl', 'wb') as fo:
        pkl.dump(res, fo)
