# Install & verify

## Prerequisites

- An **AMD GPU** supported by ROCm (developed/tested on **RX 7900 XT / gfx1100**).
- **ROCm** installed (developed on **ROCm 7.1**; `rocminfo` should list your GPU).
- A **PyTorch ROCm build** matching your ROCm, with `torch.version.hip` set.
- Python ≥ 3.10.

Check:

```bash
rocminfo | grep -m1 gfx
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
# torch.cuda.is_available() must be True; torch.version.hip must be non-None
```

## Install (editable)

```bash
git clone https://github.com/<your-username>/rocm-graph.git
cd rocm-graph
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Dependencies (`torch`, `numpy`, `scipy`) are declared in `pyproject.toml`. Install
the ROCm flavour of torch first if your environment does not already have it, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/rocm<ver>`.

## Verify

```bash
python -c "import torch, rocgraph; print(torch.__version__, torch.version.hip, rocgraph.__version__)"
pytest -q                      # smoke tests
python repro/hip_graph_replay_segfault.py 1000   # expect: replay OK
```

## Troubleshooting

- `torch.cuda.is_available()` is `False` → wrong torch build; install the ROCm
  wheel matching your ROCm version.
- Segfaults on unrelated small runs → try `HSA_OVERRIDE_GFX_VERSION` matching your
  GPU (e.g. `11.0.0` for gfx1100). Note: this does **not** fix the large-graph
  replay bug (see `rocm_hip_graph_limit.md`).
- `rocminfo` shows no GPU → check `groups`/permissions (`render`, `video`) and
  kernel driver.
