"""Device-cached constant tensors.

**Why this exists.** Building a tensor from a Python list/scalar inside a hot loop
(``x.new_tensor([...])``, ``torch.tensor([...], device=...)``) performs a
*pageable host→device copy* every step. Under CUDA/HIP graph capture that is
illegal and aborts capture with::

    RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
    capture unless the CPU tensor is pinned.

These values are constants (geometry, inertia, direction bases), so they only
need to be moved once. Cache them per ``(device, dtype, value)`` and reuse.

The returned tensor is numerically identical to ``ref.new_tensor(values)`` /
``torch.tensor(values, device=ref.device, dtype=ref.dtype)`` (same
``torch.as_tensor`` conversion), so swapping them in is behaviour-preserving.
"""
from __future__ import annotations

import torch
from torch import Tensor

_CACHE: dict[tuple, Tensor] = {}


def _key(device, dtype, values) -> tuple:
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    return (str(torch.device(device)), str(dtype), repr(values))


def const_like(ref: Tensor, values) -> Tensor:
    """Constant tensor with the same device/dtype as ``ref`` (cached)."""
    k = _key(ref.device, ref.dtype, values)
    t = _CACHE.get(k)
    if t is None:
        t = torch.as_tensor(values, dtype=ref.dtype, device=ref.device)
        _CACHE[k] = t
    return t


def const(device, dtype, values) -> Tensor:
    """Constant tensor on an explicit device/dtype (cached)."""
    k = _key(device, dtype, values)
    t = _CACHE.get(k)
    if t is None:
        t = torch.as_tensor(values, dtype=dtype, device=device)
        _CACHE[k] = t
    return t
