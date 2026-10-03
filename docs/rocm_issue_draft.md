# ROCm issue draft — HIP graph replay SIGSEGV for large graphs

> Ready-to-file report. Fill in the environment block and post to
> <https://github.com/ROCm/hip/issues> (or the relevant `ROCm/rocm-systems`
> component).

**Title:** `[Issue]: hipGraphLaunch (replay) SIGSEGV for large captured graphs`

## Summary

On ROCm 7.1 / gfx1100, a captured HIP graph **instantiates successfully** but
**`hipGraphLaunch` (replay) segfaults** once the graph is large enough. The crash
is a hard SIGSEGV inside `libamdhip64.so` (not a catchable error).

Unlike [ROCm/hip#3887](https://github.com/ROCm/hip/issues/3887) (stale device
pointers at ~200+ nodes), this is a **deterministic segfault**, and the reproducer
needs no project code.

## Environment

- GPU: AMD Radeon RX 7900 XT (gfx1100 / RDNA3)
- ROCm: 7.1  (`hip 7.1.52802`)
- PyTorch: 2.13.0+rocm7.1
- OS / kernel: _(fill in)_

## Minimal reproducer

`repro/hip_graph_replay_segfault.py`:

```python
import sys, torch
k = int(sys.argv[1])
w = torch.randn(16, 16, device="cuda", requires_grad=True)
x = torch.randn(16, 4, device="cuda")
def f():
    for _ in range(k):
        (w @ x).sum().backward()
f(); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    f()                      # capture + instantiate OK
torch.cuda.synchronize()
g.replay()                   # SIGSEGV for k >= ~2000
torch.cuda.synchronize()
print("replay OK")
```

## Observed

| `k` (matmul fwd+bwd pairs) | capture | replay |
|---:|---|---|
| 1000 | ✅ | ✅ |
| 2000 | ✅ | ❌ SIGSEGV |
| 5000 | ✅ | ❌ SIGSEGV |

Kernel log:

```
python[...]: segfault at ... ip ... in libamdhip64.so[...] likely on CPU ...
```

In a real workload the threshold is consistent with node count: a pure
forward chain captures+replays up to ~8 steps and crashes at ≥16; a
forward+loss+backward chain captures+replays up to ~4 steps and crashes at ≥8.

## Things already ruled out

- Not an operation-count issue: small graphs with the same ops are fine.
- Not host stack: `ulimit -s unlimited` has no effect.
- Not capture mode: `capture_error_mode="relaxed"` has no effect.
- Not pageable H2D / host sync: the reproducer has none inside the graph, and
  `HSA_ENABLE_SDMA=0` / `AMD_DIRECT_DISPATCH=0` do not help.

Control: `torch.compile(f, mode="reduce-overhead")` (PyTorch partitions into small
subgraphs) runs the same `k=5000` workload without crashing — suggesting the
workload itself is fine and the issue is in HIP graph execution for large graphs.

## Expected

`hipGraphLaunch` should either succeed or return an error code; it should not
segfault.

## Additional context

- The graph is a strictly linear chain of kernel nodes.
- `hipGraphInstantiate` returns success.
- Similar family: #3887, and [pytorch#155720](https://github.com/pytorch/pytorch/issues/155720).
