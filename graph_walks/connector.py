from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def temporal_rw(
    rowptr: Tensor,
    col: Tensor,
    ts: Tensor,
    start: Tensor,
    walk_length: int,
    return_edge_indices: bool = False,
    min_ts: Union[int, Tensor] = None,
    max_ts: Union[int, Tensor] = None,
    reverse: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Samples random walks of length :obj:`walk_length` from all node indices
    in :obj:`start` in the graph given by :obj:`(row, col)` s.t. each edge is
    older than the previous one sampled.

    WARNING this method does not baby you. It's just a hook into the underlying
    torch cpp/cuda implementation. It requires that the input is in CSR
    format already, and timestamps are sorted per node, e.g.
        ts[rowptr[i]]:ts[rowptr[i+1]] are strictly ascending in order
    Also (for now) timecodes have to be ints

    Args:
        row (LongTensor): Source nodes.
        col (LongTensor): Target nodes.
        ts (LongTensor): Timestamp of edges.
        start (LongTensor): Nodes from where random walks start.
        walk_length (int): The walk length.
        min_ts (int or LongTensor): Minimal value of ts to start walks at
        max_ts (int or LongTensor): Maximal value of ts to not walk past 
        return_edge_indices (bool, optional): Whether to additionally return
            the indices of edges traversed during the random walk.
            (default: :obj:`False`)

    :rtype: :class:`LongTensor`
    """
    use_discrete = (isinstance(max_ts, int) or isinstance(min_ts, int)) or (max_ts is None and min_ts is None)

    # Single min/max_ts applies to every node being traversed
    if use_discrete:
        if min_ts is None:
            min_ts = 0
        if max_ts is None:
            max_ts = 0 if not reverse else ts.max().item()

        node_seq, edge_seq = torch.ops.graph_walks.temporal_random_walk(
            rowptr, col, ts, start, walk_length, min_ts, max_ts, reverse)

    # Tensor with individual min/max_ts used per-node in the walk
    else:
        assert  (isinstance(min_ts, Tensor) or min_ts is None) and \
                (isinstance(max_ts, Tensor) or max_ts is None), \
                "`max_ts` and `min_ts` must both be Tensors, both be ints, or one or both may be `None`"

        if min_ts is None:
            min_ts = torch.zeros_like(col)
        if max_ts is None:
            val = 0 if not reverse else ts.max().item()
            max_ts = torch.full_like(col, val)

        node_seq, edge_seq = torch.ops.graph_walks.continuous_trw(
            rowptr, col, ts, start, walk_length, min_ts, max_ts, reverse)

    if return_edge_indices:
        return node_seq, edge_seq

    return node_seq

def rw(rowptr: Tensor,
    col: Tensor,
    start: Tensor,
    walk_length: int,
    return_edge_indices: bool = False,
    p = 1.0, q = 1.0 
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Samples random walks of length :obj:`walk_length` from all node indices
    in :obj:`start` in the graph given by :obj:`(row, col)`

    Clone of old torch-cluster rw function that's no longer maintained 
    Necessitated because Pyg-Lib RW does not return edge indices 

    Args:
        rowptr (LongTensor): Source nodes index pointer.
        col (LongTensor): Target nodes.
        start (LongTensor): Nodes from where random walks start.
        walk_length (int): The walk length.
        return_edge_indices (bool, optional): Whether to additionally return
            the indices of edges traversed during the random walk.
            (default: :obj:`False`)

    :rtype: :class:`LongTensor`
    """
    node_seq, edge_seq = torch.ops.graph_walks.random_walk(rowptr, col, start, walk_length, p,q)

    if return_edge_indices:
        return node_seq, edge_seq

    return node_seq