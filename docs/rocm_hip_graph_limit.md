# Is this the ROCm limit? Hardware vs driver vs patchability

Measured on **RX 7900 XT (gfx1100 / RDNA3)**, **ROCm 7.1**, **torch 2.13.0+rocm7.1**.

## 1. Is it a ROCm limit?

It is a **software (HIP runtime / driver) defect**, not a hardware limit. Evidence:

- Minimal 1-kernel graphs always work; the failure appears **only as graph size
  (node count) grows**.
- Capture **and** `hipGraphInstantiate` succeed; the crash is in
  `hipGraphLaunch` (replay), a hard SIGSEGV inside `libamdhip64.so`.
- Same-family reported issues:
  - [ROCm/hip#3887](https://github.com/ROCm/hip/issues/3887) — stale device-pointer
    reads at ~200+ nodes (RDNA4 / gfx1201).
  - [pytorch#155720](https://github.com/pytorch/pytorch/issues/155720) — HIP graph
    capture segfault.
  - [pytorch#187715](https://github.com/pytorch/pytorch/issues/187715) —
    `reduce-overhead` has no HIP guard and degrades training on ROCm.

## 2. What the hardware supports

`rocminfo` for gfx1100: **Max Queue Size 131072 packets**, Grid Max 2^32-1, 64 CU.
HIP graphs are lowered to command buffers / AQL packets handed to the command
processor. There is **no hardware limit near ~2000 nodes**; a 64-step graph
(~10^5 kernels) is within the command processor's capabilities.

## 3. Is the driver doing a bad job?

Yes — the failure mode is a **segfault** (driver writes to an invalid address),
not a clean error code. The usual workarounds do **not** help:
`HSA_ENABLE_SDMA=0`, `AMD_DIRECT_DISPATCH=0`, `capture_error_mode="relaxed"`,
`ulimit -s unlimited`.

## 4. Can it be patched?

Not as a "small patch": the HIP runtime is open source, but fixing it means
locating the out-of-bounds / stale-pointer path in `libamdhip64.so`, changing the
runtime, and **rebuilding the whole ROCm stack** (large, ABI-sensitive). Not
practical from the application side.

**Realistic workarounds (no driver changes):**

| Workaround | Result |
|---|---|
| `torch.compile(mode="reduce-overhead")` — cudagraph trees automatically partition into small subgraphs | **Does not crash** (K=5000 works where manual single-graph capture segfaults). Applying it to a *recurrent* training loop still requires the static-buffer refactor (cudagraph trees treat recurrent state as outputs). |
| Split a large graph into subgraphs ≤ threshold and replay them in a chain | Works, but shrinks the backward-through-time horizon to the subgraph length. |
| File a ROCm issue | Recommended; the reproducer here is deterministic (better than #3887). |

## See also

- `repro/hip_graph_replay_segfault.py` — minimal standalone reproducer.
- `docs/rocm_issue_draft.md` — ready-to-file report.
