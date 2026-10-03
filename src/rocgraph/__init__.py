"""rocgraph — capture-safe HIP/CUDA graph pieces for recurrent training on AMD ROCm.

Public helpers:

* :func:`rocgraph.const_like` / :func:`rocgraph.const` — device-cached constant
  tensors (building a tensor from a Python list inside a hot loop is a hidden,
  non-obvious graph-capture killer: it is a pageable H2D copy).
* :func:`rocgraph.build_segment_sum_structure` /
  :func:`rocgraph.segment_sum_spmm` — a capture-safe sparse-matmul backend that
  replaces ``torch.sparse.mm`` (whose rocsparse *preprocess* stage is not
  graph-capturable).

See the repository README for the full write-up and the ROCm HIP-graph defects
this works around.
"""

from __future__ import annotations

from .const_cache import const, const_like
from .segment_sum import (
    SegmentSumStructure,
    build_segment_sum_structure,
    segment_sum_spmm,
)

__all__ = [
    "const",
    "const_like",
    "SegmentSumStructure",
    "build_segment_sum_structure",
    "segment_sum_spmm",
]
__version__ = "0.0.1"
