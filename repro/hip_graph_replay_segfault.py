#!/usr/bin/env python
"""Minimal reproducer: HIP graph replay segfaults above a graph-size threshold.

    python repro/hip_graph_replay_segfault.py [K]

K = number of matmul fwd+bwd pairs placed in a single captured graph
(roughly K*~3 kernel nodes). ``hipGraphInstantiate`` succeeds; ``hipGraphLaunch``
(replay) crashes.

Measured on RX 7900 XT (gfx1100 / RDNA3), ROCm 7.1, torch 2.13.0+rocm7.1:

    K=1000  -> replay OK
    K>=2000 -> SIGSEGV (exit 139), crash inside libamdhip64.so

Control: ``torch.compile(f, mode="reduce-overhead")`` (which partitions into
small subgraphs) runs the same K=5000 workload without crashing — i.e. this is a
HIP-graph *runtime* defect, not a hardware limit.
"""
import sys

import torch


def main() -> int:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    dev = "cuda"
    w = torch.randn(16, 16, device=dev, requires_grad=True)
    x = torch.randn(16, 4, device=dev)

    def f() -> None:
        for _ in range(k):
            (w @ x).sum().backward()

    f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    torch.cuda.synchronize()
    print(f"capture + instantiate ok (K={k}); replaying...", flush=True)
    g.replay()  # <-- SIGSEGV here for K >= ~2000
    torch.cuda.synchronize()
    print(f"replay OK (K={k})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
