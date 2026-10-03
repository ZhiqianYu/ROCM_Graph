# Testing

Three layers: **reproducers** (prove the ROCm bugs), **unit tests** (correctness of
the building blocks), **benchmarks** (speed + numeric equivalence).

All commands assume the venv is active and you are in the repo root.

---

## 1. Reproducers (no project code needed)

```bash
python repro/hip_graph_replay_segfault.py 1000   # expect: "replay OK"
python repro/hip_graph_replay_segfault.py 2000   # expect: SIGSEGV (exit 139)
```

Interpretation: capture + `hipGraphInstantiate` succeed; `hipGraphLaunch` (replay)
crashes above the size threshold. This is the ROCm defect this project works
around. On a *non-buggy* runtime both should print `replay OK`.

---

## 2. Unit tests

```bash
pytest -q
```

`tests/` covers:
- `const_cache`: same device/dtype, value equality, cache reuse.
- `segment_sum`: shape/dtype, correct `W @ x` vs a dense reference, capture under
  a `torch.cuda.CUDAGraph` (skipped if no GPU/ROCm).

Run a single file: `pytest -q tests/test_smoke.py`.

---

## 3. Benchmark / numeric equivalence

The reference comparison is **`segment_sum` vs `torch.sparse.mm`** (rocsparse), and
**eager vs graph-captured** chunks.

```bash
# (planned) benchmark script
python benchmarks/bench_spmm.py        # ms/call: rocsparse vs segment_sum
python benchmarks/bench_chunk.py       # ms/chunk: eager vs reduce-overhead, + grad equality
```

Expected numbers (RX 7900 XT / ROCm 7.1, for reference — measure your own):
- `segment_sum` SpMM ≈ 0.4 ms vs rocsparse ≈ 0.2 ms per call (brain is a small part
  of the step, so this is ≈ 2–4% of step time).
- `segment_sum` vs `sparse.mm`: max|Δ| ≈ 1 ULP, **not** bit-identical, **deterministic**.
- capture speed-up ceiling = `wall / gpu_busy` ≈ **1.97×**; target ≈ **1.9×**.

---

## 4. How to interpret results

- **Correctness first.** A captured path is only usable if
  `torch.equal(grad_eager, grad_captured)` (or the project re-baselines and
  documents the difference).
- **Numeric drift is expected** for `segment_sum` (1 ULP) — record it, do not
  claim bit-identity.
- **A segfault is never "your bug" alone** if it's in `libamdhip64.so` on replay of
  a large graph — that's the ROCm defect; keep graphs small.
