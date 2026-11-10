'''
Checks the time bounds of temporal_rw against brute force on a small random graph.

For every start node and every combination of (forward | reverse) x (int | tensor bounds):

  1. every edge a walk takes lies inside [min_ts, max_ts];
  2. timestamps never decrease along a forward walk and never increase along a reverse walk;
  3. over many samples, the set of FIRST edges taken from a node is exactly the set of its
     out-edges inside the window. This is the check the old binary_search_max failed: it never
     sampled the latest edge in the window, and could sample one after max_ts.

Runs on CPU, and on CUDA as well when it is available. A few seconds; needs only torch and the
built extension.

    python test/test_time_bounds.py
'''
import torch

from graph_walks import temporal_rw


def random_csr(num_nodes, num_edges, max_t, gen):
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    ts = torch.randint(0, max_t + 1, (num_edges,), generator=gen)
    # sort by (src, ts): rows in node order, timestamps ascending inside each row
    order = torch.argsort(src * (max_t + 1) + ts, stable=True)
    src, dst, ts = src[order], dst[order], ts[order]
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.bincount(src, minlength=num_nodes).cumsum(0)
    return rowptr, dst, ts


def check(device, reverse, tensor_bounds, gen, samples=400, walk_len=4):
    n, max_t = 40, 30
    rowptr, col, ts = random_csr(n, 300, max_t, gen)
    lo = torch.randint(0, max_t, (n,), generator=gen)
    # hi >= 1 always: for forward walks the library reads max_ts == 0 as 'no upper bound'
    hi = (lo + torch.randint(1, 12, (n,), generator=gen)).clamp(max=max_t)
    if not tensor_bounds:                       # one window for everybody; keep both bounds non-zero
        lo[:] = 5
        hi[:] = 17

    start = torch.arange(n).repeat(samples)
    lo_w, hi_w = lo.repeat(samples), hi.repeat(samples)
    d = lambda x: x.to(device)
    kw = dict(min_ts=d(lo_w), max_ts=d(hi_w)) if tensor_bounds else dict(min_ts=5, max_ts=17)
    walks, eids = temporal_rw(d(rowptr), d(col), d(ts), d(start), walk_len,
                              return_edge_indices=True, reverse=reverse, **kw)
    walks, eids = walks.cpu(), eids.cpu()
    taken = eids >= 0
    t = ts[eids.clamp(min=0)]

    # 1. inside the window
    assert (t[taken] >= lo_w.unsqueeze(1).expand_as(t)[taken]).all(), 'edge before min_ts'
    assert (t[taken] <= hi_w.unsqueeze(1).expand_as(t)[taken]).all(), 'edge after max_ts'

    # 2. monotone in time, and a walk that stopped stays stopped
    both = taken[:, 1:] & taken[:, :-1]
    step = t[:, 1:] - t[:, :-1]
    assert ((step <= 0) if reverse else (step >= 0))[both].all(), 'timestamps out of order'
    assert not (taken[:, 1:] & ~taken[:, :-1]).any(), 'walk resumed after it had stopped'

    # 3. first hop covers exactly the in-window out-edges of each start node
    for v in range(n):
        row = torch.arange(int(rowptr[v]), int(rowptr[v + 1]))
        want = set(row[(ts[row] >= lo[v]) & (ts[row] <= hi[v])].tolist())
        got = set(eids[start == v, 0].tolist()) - {-1}
        assert got <= want, f'node {v}: sampled edges outside its window: {sorted(got - want)}'
        # 400 draws from at most a few dozen edges: missing one by chance is ~1e-5 or less
        assert got == want, f'node {v}: in-window edges never sampled: {sorted(want - got)}'
        assert (eids[start == v, 0] == -1).all() == (len(want) == 0)


if __name__ == '__main__':
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        for reverse in (False, True):
            for tensor_bounds in (False, True):
                gen = torch.Generator().manual_seed(0)
                check(device, reverse, tensor_bounds, gen)
                print(f'ok  {device:<4}  {"reverse" if reverse else "forward"}  '
                      f'{"tensor" if tensor_bounds else "int"} bounds')
    print('all time-bound checks passed')
