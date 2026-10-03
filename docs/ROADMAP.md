# Roadmap

Milestones with acceptance criteria. Update this file as work lands.

---

## M0 — Scaffold ✅ (2026-10-03)

- Repo, license (PolyForm NC 1.0.0), citation, install docs, handoff.
- **Acceptance:** `pip install -e .` works; `pytest` collects.

## M1 — Capture-safe SpMM ✅ (2026-10-03)

- `src/rocgraph/segment_sum.py`; `src/rocgraph/const_cache.py`.
- **Acceptance (met):** segment_sum captures under HIP graph on gfx1100; deterministic;
  ~1 ULP from rocsparse (documented, not bit-identical).

## M2 — Static-buffer recurrent training helper ⏳ (next)

- Provide helpers to keep recurrent state (ledger/body/brain/per-channel filter
  state) in **static buffers updated in place**, and to run a captured
  `fwd+loss+bwd` chunk with `torch.compile(mode="reduce-overhead")`.
- **Acceptance:**
  1. A recurrent chunk runs ≥10 iterations under `reduce-overhead` without the
     `accessing tensor output of CUDAGraphs that has been overwritten` error.
  2. `torch.equal(loss_eager, loss_captured)` and per-parameter
     `torch.equal(grad_eager, grad_captured)` (same parameter snapshot).
  3. Default (non-captured) path unchanged.

## M3 — Benchmark & report ⏳

- `benchmarks/bench_spmm.py`, `benchmarks/bench_chunk.py`.
- **Acceptance:** report `ms/chunk` eager vs captured and the ratio (target ≈1.9×),
  on gfx1100; publish numbers in `docs/`.

## M4 — Tests & CI ⏳

- Expand `tests/`; a GPU-marked test that captures a real chunk.
- **Acceptance:** `pytest -q` green on CPU (GPU tests skipped) and on ROCm.

## M5 — Upstream the ROCm bug ⏳

- File `docs/rocm_issue_draft.md`; link the issue here.

## Out of scope (for now)

- Rewriting the runtime / patching `libamdhip64.so` (not practical).
- Non-ROCm platforms (but CUDA should work; `rocgraph` is device-agnostic where possible).
