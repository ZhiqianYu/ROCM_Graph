"""Capture-safe sparse matmul (``y = W @ x``) via ``F.embedding_bag`` segment-sum.

**Why.** ``torch.sparse.mm`` on ROCm goes through ``rocsparse_spmm``, whose
*preprocess* stage is not supported inside HIP graphs, so a graph containing a
sparse matmul fails to capture. This module expresses SpMM as a per-output-row
segment sum using ``F.embedding_bag(mode="sum")``, which *is* capturable and
deterministic.

Given a canonical CSR matrix ``W`` (shape ``(N, N)``, values supplied per call),
the output is

    ``y[b, i] = Σ_{e : row_e = i} values_e · x[b, col_e]``

built from the static structure (``col``, ``offsets``, ``arange``) exported once
from the CSR's row-major COO expansion.

Notes
-----
* This is **semantically equivalent** to ``torch.sparse.mm`` but **not
  bit-identical** (rocsparse uses fused multiply-add; ``embedding_bag`` rounds the
  products first) — differences are ~1 ULP. It is *deterministic*.
* ``rocsparse`` itself remains the default in the host project; this backend is
  opt-in for the graph-capture path.
"""
from __future__ import annotations

from dataclasses import dataclass
import weakref

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SegmentSumStructure:
    """Static, device-resident structure derived from a canonical CSR matrix."""

    col: Tensor       # (nnz,) source (pre-synaptic / column) indices, row-major
    offsets: Tensor   # (N+1,) bag boundaries, last element == nnz
    arange: Tensor    # (nnz,) indices for embedding_bag
    shape: tuple[int, int]
    n_neurons: int


_CACHE: dict[tuple, dict] = {}


def build_segment_sum_structure(csr, device: torch.device) -> SegmentSumStructure:
    """Export the static structure from a SciPy CSR matrix (row-major, canonical)."""
    key = (id(csr), torch.device(device))
    ent = _CACHE.get(key)
    if ent is not None and ent["csr"]() is csr:
        return ent["struct"]
    if not (csr.has_sorted_indices and csr.has_canonical_format):
        csr = csr.copy()
        csr.sum_duplicates()
        csr.sort_indices()
    coo = csr.tocoo()
    row = np.asarray(coo.row)
    col = np.asarray(coo.col)
    n = int(csr.shape[0])
    counts = np.bincount(row, minlength=n).astype(np.int64)
    offs = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offs[1:])
    assert int(offs[-1]) == int(col.size)
    struct = SegmentSumStructure(
        col=torch.as_tensor(col, dtype=torch.long, device=device),
        offsets=torch.as_tensor(offs, dtype=torch.long, device=device),
        arange=torch.arange(col.size, dtype=torch.long, device=device),
        shape=(n, n), n_neurons=n,
    )
    _CACHE[key] = {"csr": weakref.ref(csr), "struct": struct}
    return struct


def segment_sum_spmm(values: Tensor, x: Tensor, struct: SegmentSumStructure) -> Tensor:
    """``y = W @ x`` with non-zero values ``values`` (same order as ``struct.col``).

    Args:
      values: ``(nnz,)`` non-zero values of ``W`` in canonical row-major order.
      x: ``(B, N)`` states (or ``(N,)``). Returns ``(B, N)`` (or ``(N,)``).
    """
    one_d = x.dim() == 1
    x2 = x.reshape(1, -1) if one_d else x
    contrib = values.unsqueeze(1) * x2.t()[struct.col]          # (nnz, B)
    y = F.embedding_bag(struct.arange, contrib, struct.offsets,
                        mode="sum", include_last_offset=True)   # (N, B)
    y = y.t()                                                   # (B, N)
    return y.reshape(-1) if one_d else y
