# rocm-graph-train

**Making large HIP/CUDA-graph capture actually work for recurrent, autograd-based
training on AMD ROCm.**

> Status: **early / WIP.** This project is being built out of a concrete need:
> training a recurrent, differentiable simulator (a spiking connectome-driven
> controller) on an RX 7900 XT (gfx1100 / RDNA3), where the workload is
> **launch-bound** (≈2,600 kernels/step, GPU only ~24% TDP) and the obvious fix —
> graph capture — is blocked by ROCm bugs.

---

## The problem

Graph capture is the standard cure for launch-bound training/inference: record the
stream once, replay the whole DAG with one launch. On AMD ROCm it breaks in three
independent, reproducible ways:

1. **`torch.sparse.mm` (rocsparse) is not capturable.**
   `rocsparse_spmm`'s *preprocess* stage is not supported inside HIP graphs, so any
   graph containing a sparse matmul fails to capture (`hipErrorStreamCaptureInvalidated`).

2. **Large graphs segfault on replay.**
   HIP graph **capture + instantiation succeed**, but `hipGraphLaunch` (replay)
   raises SIGSEGV inside `libamdhip64.so` above a graph-size threshold.
   Minimal repro (no project code needed):

   ```bash
   python repro/hip_graph_replay_segfault.py 2000   # SIGSEGV
   python repro/hip_graph_replay_segfault.py 1000   # OK
   ```

   This is a **driver/runtime defect**, not a hardware limit: the command processor
   handles ≥131072 packets, and `torch.compile(mode="reduce-overhead")` (which splits
   into small subgraphs) runs the same workload fine.
   Related: [ROCm/hip#3887](https://github.com/ROCm/hip/issues/3887),
   [pytorch#155720](https://github.com/pytorch/pytorch/issues/155720).

3. **`torch.compile(mode="reduce-overhead")` has no HIP guard.**
   cudagraph trees silently mis-handle (or error on) recurrent state on ROCm —
   see [pytorch#187715](https://github.com/pytorch/pytorch/issues/187715).
   Naïvely applying it to a recurrent training loop raises
   `accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`.

## What this project aims to provide

- **`segment_sum` SpMM** — a capture-safe replacement for `torch.sparse.mm`
  (`F.embedding_bag` over a canonical-CSR-derived static structure). Semantically
  equivalent, deterministic, graph-capturable.
- **Device-cached constants** — a tiny helper so hot loops never build tensors from
  Python lists (a hidden, non-obvious capture killer).
- **Static-buffer recurrent training** — the refactor required to make captured
  recurrent/autograd chunks correct (no address drift, no escaping tensors).
- **Workarounds + minimal repros** for the ROCm HIP-graph bugs above.
- **Benchmarks** (eager vs captured, wall-clock + numeric equivalence) on gfx1100.

The goal is a small, dependency-light toolkit + a write-up that lets anyone else on
ROCm reproduce the problem and apply the fix.

## Repository layout

```text
repro/        minimal, standalone reproducers for the ROCm bugs
src/rocgraph/ the capture-safe pieces (SpMM, const-cache, capture helpers)
docs/         the analysis, the ROCm issue draft, findings
```

## License & citation

This project is **source-available** (not OSI "open source"):

- **Personal, academic, and other non-commercial use: free**, but **you must cite**
  this work (see [`CITATION.cff`](CITATION.cff)).
- **Commercial use requires a separate license** from the copyright holder.
- Governed by the **PolyForm Noncommercial License 1.0.0** ([`LICENSE`](LICENSE)).

> ⚠️ Because commercial use is restricted, this is *not* an OSI-approved open-source
> license. It is a **fair-source / source-available** license.
