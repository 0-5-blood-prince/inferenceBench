# Environment findings from this pod

- **Runpod A100 volumes are network storage (MooseFS).** Measured: 800 small
  file writes took 184 ms on local NVMe and 15.6 s on the volume — **85x**;
  sequential only 4.7x. Venvs on local disk, weights on the volume. Turned a
  45-minute bootstrap into ~4 minutes.
- **No A100 with a CUDA 13 driver was available** (best 570 / CUDA 12.8;
  requiring 13.0 returned "no instances available"), but both engines pull torch
  cu130. Fixed with `cuda-compat-13-0` (580.178.04) ahead of the system libcuda
  — the supported forward-compat path on datacenter GPUs.
- **GPU clocks cannot be locked from inside the container**, so SPEC's
  thermal-drift control is unavailable. Mitigation: P3's engine-order
  alternation and the variance-duplicate cell.
- **`ninja` is required but undeclared** by both engines — torch.compile shells
  out to it, so without it the server dies *after* loading 60 GB of weights.
- **SGLang serves no Prometheus endpoint without `--enable-metrics`**, and no
  per-request cache fields without `--enable-cache-report`.
- **Gemma 4 uses a fixed 258 image tokens at every resolution** (448², 896²,
  1344²) — not a dynamic-resolution ViT, so SPEC §5's resolution-fixing
  rationale does not apply. Resolution was chosen instead to minimise
  client-side base64 payload. (This constant is what makes the image/text
  token split in
  [../measurement/metrics-instrumentation.md](../measurement/metrics-instrumentation.md)
  possible on an engine that never reports it directly.)
- **SGLang reports `max_total_num_tokens = 16391`** — only ~16 requests of KV at
  our T=991, which will shape the saturation knee (see
  [../measurement/pilot-methodology.md](../measurement/pilot-methodology.md))
  and make eviction bite early.

Related: [why CUDA graph capture behaviour is itself a recorded, not equalized,
engine default »](cuda-graphs.md)
