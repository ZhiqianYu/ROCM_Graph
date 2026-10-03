# HANDOFF — start here (new session)

> **One-page entry point.** Read this first; it tells you the mission, what is
> done, what to do next (with commands), how to test, and the hard constraints.
> Generated 2026-10-03 for `v0.0.1`.

---

## 0. One-line status

**Scaffold + two capture-safe building blocks (`segment_sum` SpMM, `const_cache`)
exist; the ROCm HIP-graph limits are measured and reproduced; the static-buffer
recurrent-training integration is the next thing to build.**

---

## 1. Mission

Remove the launch overhead that dominates training of **launch-bound recurrent /
differentiable-simulator** workloads on AMD ROCm, by making **HIP/CUDA graph
capture** usable. Concretely: provide capture-safe building blocks + workarounds
for three ROCm defects (sparse-matmul capture, large-graph replay segfault,
cudagraph-trees-on-ROCm), and document them so others can reproduce and reuse.

Why it matters: on gfx1100 the reference workload runs at **GPU ~24% TDP** with
**≈2.6k kernels/step** — the GPU is idle waiting on launches. Filling that idle is
worth **~1.97×** (`docs/rocm_hip_graph_limit.md`). Graph capture is the direct way
to realize it.

---

## 2. What exists now

| Path | What |
|---|---|
| `src/rocgraph/segment_sum.py` | **capture-safe SpMM** via `F.embedding_bag` segment-sum. Semantically equivalent to `torch.sparse.mm`, **deterministic**, capturable; **~1 ULP** different (not bit-identical). |
| `src/rocgraph/const_cache.py` | `const_like` / `const` — device-cached constants; kills the hidden pageable-H2D capture killer. |
| `repro/hip_graph_replay_segfault.py` | standalone minimal reproducer of the ROCm large-graph replay SIGSEGV. |
| `docs/rocm_hip_graph_limit.md` | hardware-vs-driver analysis (arguments that it is a driver defect, not hardware). |
| `docs/rocm_issue_draft.md` | ready-to-file ROCm issue (deterministic repro). |

**Proven (measured on RX 7900 XT / ROCm 7.1 / torch 2.13):**
- `torch.sparse.mm` capture → `hipErrorStreamCaptureInvalidated`; `segment_sum` capture → **OK**.
- HIP graph replay SIGSEGV: `repro` K≤1000 OK, K≥2000 crash; forward-only chain crashes at ≥16 steps, fwd+bwd at ≥8.
- `torch.compile(mode="reduce-overhead")` **does not crash** at K=5000 (it partitions into small subgraphs) — but needs the static-buffer refactor for recurrent state.

---

## 3. What to do next (priority order)

### P0 — Reproduce / verify the environment (5 min)
```bash
python repro/hip_graph_replay_segfault.py 1000   # expect: replay OK
python repro/hip_graph_replay_segfault.py 2000   # expect: SIGSEGV (exit 139)
```

### P1 — Static-buffer recurrent training helper (the main missing piece)
Make captured recurrent chunks **correct** under cudagraph trees: recurrent state
(energy ledger, body state, brain state, per-channel filter state) must live in
**static buffers updated in place** (`buf.copy_()`), never re-bound, and the
captured function must not leak tensors. See `docs/ROADMAP.md` M2 for the spec and
acceptance criteria.

### P2 — cudagraph-trees integration + benchmark
Wrap a `fwd+loss+bwd` chunk with `torch.compile(mode="reduce-overhead")`, verify
`torch.equal(grad_eager, grad_captured)`, and measure `ms/chunk` (target ≈1.9×).

### P3 — Tests & CI
Extend `tests/` (see `docs/TESTING.md`); add a GPU benchmark script.

### P4 — Publish the ROCm issue
Post `docs/rocm_issue_draft.md` to `ROCm/hip` (or `ROCm/rocm-systems`).

---

## 4. Hard constraints / gotchas

- **Not all ROCm ops are capturable.** `torch.sparse.mm` (rocsparse preprocess),
  `torch.linalg.solve`, and host syncs / pageable H2D are common blockers. Prefer
  small, capture-friendly equivalents (see `segment_sum`, `const_cache`).
- **Large HIP graphs segfault on replay** (driver defect). Keep any *single* graph
  small; use `reduce-overhead` (auto-partition) rather than one huge manual graph.
- **`reduce-overhead` on ROCm is under-supported** (pytorch#187715). Verify
  numerics; it can silently degrade.
- **`index_add` / `scatter_add` are capture-safe but non-deterministic** (atomic
  adds). Use `embedding_bag` (deterministic) for SpMM.
- **Numerics:** `segment_sum` is ~1 ULP from rocsparse, so results are **not
  bit-identical** — a project using it must re-baseline.

---

## 5. Where things are / how to test

| Need | Go to |
|---|---|
| Install & verify | `docs/INSTALL.md` |
| Run tests / repro / benchmarks | `docs/TESTING.md` |
| Milestones & acceptance criteria | `docs/ROADMAP.md` |
| Why it's a driver bug | `docs/rocm_hip_graph_limit.md` |
| File the ROCm bug | `docs/rocm_issue_draft.md` |

---

## 6. Provenance

Extracted from a research project (`BrainSimu`) that trains a spiking,
connectome-driven controller. The capture path was bisected there; the generic
pieces were lifted here. The reference "real workload" is a 64-step TBPTT chunk of
a recurrent brain + body + world, which is launch-bound.

## 7. License

Source-available: **PolyForm Noncommercial 1.0.0** (`LICENSE`). Non-commercial free
**with citation** (`CITATION.cff`); **commercial requires a separate license**.
Not OSI "open source".
