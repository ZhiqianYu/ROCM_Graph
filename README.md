# rocm-graph

**Make large HIP/CUDA graph capture work for launch-bound training on AMD ROCm.**

> Status: **early / WIP** (`v0.0.1`). Born from a concrete need: training a
> recurrent, differentiable simulator on an **RX 7900 XT (gfx1100 / RDNA3)** where
> the workload is **launch-bound** (≈2.6k kernels/step, GPU ~24% TDP). Graph
> capture — the standard cure — is blocked by ROCm bugs. This project collects the
> workarounds, minimal reproducers, and the capture-safe building blocks.

> **New here, or starting a fresh session?** Read **[`docs/HANDOFF.md`](docs/HANDOFF.md)**
> first — it is the one-page entry point (mission, status, next steps, constraints).
> Then see [Install](#install) and [`docs/TESTING.md`](docs/TESTING.md).

---

## 1. The problem: three reproducible ROCm HIP-graph blockers

1. **`torch.sparse.mm` (rocsparse) is not capturable.**
   `rocsparse_spmm`'s *preprocess* stage is unsupported inside HIP graphs, so any
   captured graph containing a sparse matmul fails (`hipErrorStreamCaptureInvalidated`).

2. **Large graphs SIGSEGV on replay.**
   HIP graph **capture and `hipGraphInstantiate` succeed**, but `hipGraphLaunch`
   (replay) crashes inside `libamdhip64.so` above a graph-size threshold. Minimal,
   project-independent reproducer:

   ```bash
   python repro/hip_graph_replay_segfault.py 1000   # replay OK
   python repro/hip_graph_replay_segfault.py 2000   # SIGSEGV (exit 139)
   ```

   This is a **driver/runtime defect, not a hardware limit** — the gfx1100 command
   processor handles ≥131072 packets, and `torch.compile(mode="reduce-overhead")`
   (which partitions into small subgraphs) runs the same workload fine.
   Related: [ROCm/hip#3887](https://github.com/ROCm/hip/issues/3887),
   [pytorch#155720](https://github.com/pytorch/pytorch/issues/155720).

3. **`torch.compile(mode="reduce-overhead")` has no HIP guard.**
   cudagraph trees silently mis-handle recurrent state on ROCm
   ([pytorch#187715](https://github.com/pytorch/pytorch/issues/187715)):
   `accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`.

## 2. What this project provides

| Piece | File | Purpose |
|---|---|---|
| **capture-safe SpMM** | `src/rocgraph/segment_sum.py` | replaces `torch.sparse.mm` with an `F.embedding_bag` segment-sum (capturable + deterministic; ~1 ULP vs rocsparse, not bit-identical) |
| **device-cached constants** | `src/rocgraph/const_cache.py` | kill the hidden pageable-H2D killer (`new_tensor([...])` in hot loops) |
| **minimal reproducers** | `repro/` | prove the ROCm bugs without any project code |
| **analysis + issue drafts** | `docs/` | hardware-vs-driver analysis, ready-to-file ROCm issue |

Planned (see [`docs/ROADMAP.md`](docs/ROADMAP.md)): static-buffer recurrent-training
helper, cudagraph-trees integration, benchmarks on gfx1100.

## 3. Install

Requires an AMD GPU with ROCm and a matching PyTorch (`torch.cuda.is_available()`
must be `True`; `torch.version.hip` set). See [`docs/INSTALL.md`](docs/INSTALL.md).

```bash
git clone https://github.com/ZhiqianYu/ROCM_Graph.git
cd ROCM_Graph
python -m venv .venv && source .venv/bin/activate
pip install -e .
# verify
python -c "import torch, rocgraph; print(torch.__version__, torch.version.hip, rocgraph.__version__)"
```

## 4. Quickstart

```python
import torch
from rocgraph import const_like, build_segment_sum_structure, segment_sum_spmm

# Device-cached constant (safe inside graph capture):
base = torch.zeros(3, device="cuda")
side = const_like(base, [-1.0, 1.0])

# Capture-safe sparse matmul, y = W @ x:
#   csr: a SciPy CSR matrix (canonical) describing W's topology
#   values: (nnz,) non-zero values, in canonical row-major order
struct = build_segment_sum_structure(csr, torch.device("cuda"))
y = segment_sum_spmm(values, x, struct)          # (B, N)
```

## 5. Repository layout

```text
docs/         HANDOFF (start here) · INSTALL · TESTING · ROADMAP · analysis · issue draft
src/rocgraph/ const_cache · segment_sum
repro/        standalone ROCm bug reproducers
tests/        smoke tests
```

## 6. License & citation

**Source-available — not OSI "open source".** Governed by the
[PolyForm Noncommercial License 1.0.0](LICENSE):

- **Personal, academic, and other non-commercial use: free**, but you **must cite**
  this work — see [`CITATION.cff`](CITATION.cff).
- **Commercial use requires a separate license** from the copyright holder.
