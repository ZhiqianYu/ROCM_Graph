"""Smoke tests — runnable on CPU (GPU tests skip if unavailable)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from rocgraph import (
    build_segment_sum_structure,
    const,
    const_like,
    segment_sum_spmm,
)


def test_const_cache_is_cached_and_correct():
    base = torch.zeros(3)
    a = const_like(base, [-1.0, 1.0])
    b = const_like(base, [-1.0, 1.0])
    assert a is b
    assert torch.equal(a, torch.tensor([-1.0, 1.0]))
    assert const("cpu", torch.float32, [1.0, 2.0]).tolist() == [1.0, 2.0]


def _segsum_setup(n: int = 24):
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((n, n))
    dense[np.abs(dense) < 1.0] = 0.0            # make it sparse
    csr = csr_matrix(dense.astype(np.float32))
    csr.sum_duplicates()
    csr.sort_indices()
    coo = csr.tocoo()                            # canonical row-major
    values = torch.as_tensor(coo.data, dtype=torch.float32)
    struct = build_segment_sum_structure(csr, torch.device("cpu"))
    return dense.astype(np.float32), values, struct, n


def test_segment_sum_matches_dense_matmul():
    dense, values, struct, n = _segsum_setup()
    x = torch.randn(4, n)
    y = segment_sum_spmm(values, x, struct)      # (B, N)
    ref = x @ torch.as_tensor(dense).t()         # y[b,i] = Σ_j x[b,j] W[i,j]
    assert y.shape == (4, n)
    assert torch.allclose(y, ref, atol=1e-4), (y - ref).abs().max().item()


def test_segment_sum_one_dimensional_input():
    dense, values, struct, n = _segsum_setup()
    x = torch.randn(n)
    y = segment_sum_spmm(values, x, struct)
    assert y.shape == (n,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_segment_sum_is_graph_capturable():
    dev = torch.device("cuda")
    dense, values, struct, n = _segsum_setup()
    values = values.to(dev)
    struct = build_segment_sum_structure(
        csr_matrix(dense).astype(np.float32), dev
    )
    x = torch.randn(4, n, device=dev)

    def run():
        return segment_sum_spmm(values, x, struct)

    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = run()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    assert y.shape == (4, n)
