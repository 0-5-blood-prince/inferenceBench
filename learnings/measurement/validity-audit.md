# Validity audit — the mechanism investigation

Every experiment in the "why is SGLang's prefill slower" chain, checked for the
things that could invalidate it: measurement isolation, confounds (cache, warmup,
JIT, thermal, engine order), config fidelity (did the flag actually engage),
comparison fairness, statistics, and — for the kernel bench — numerical
correctness. One gap found and in remediation; everything else holds.

## 1. Prefill-vs-queue decomposition (`decomp.sh`, conc=1)
- **Isolation:** `--rate 0 --concurrency 1` → one request in flight, so
  TTFT(conc=1) ≈ prefill compute with no queue. Queue = loaded − conc1 gave
  **symmetric ~70/85 ms both engines** — a sane, self-consistent result. ✓
- **Warmup:** `--warmup 10`, post-warmup requests only. ✓
- **Data hygiene:** all 26 conc=1 result files tagged `ok` — no JIT contamination,
  no errors. ✓
- **Caveat (documented, not a defect):** queue = loaded − conc1 assumes per-request
  prefill compute is the same under load as at conc=1; batching/chunked-prefill can
  shift it slightly. It's an approximation, and the components are consistent with
  the loaded numbers. **Valid.**

## 2. Config sweep (`config_test.sh`, `prefill_graph_test.sh`)
- **torch.compile:** engaged (`enable_torch_compile=True`, `torch_compile_max_bs=32`
  in server args); it targets decode, so prefill null is expected and real. ✓
- **prefill graph (eager tc_piecewise):** capture **confirmed engaged** (startup
  log "Capture target prefill CUDA graph begin/end", 231 s). Not a silent no-op. ✓
- **flashinfer / fa3:** server **rejected** them at startup (Gemma-4 accepts only
  trtllm_mha/triton/intel_xpu) — correctly recorded as "rejected", never as a
  fabricated number. ✓ **Valid.**

## 3. Inductor-compiled prefill (`inductor_prefill_test.sh`)
- **Engagement:** startup capture shows `tc_compiler='inductor'`, 414 s compile
  (vs 231 s eager) — Inductor genuinely ran. ✓
- **No eager fallback for the 991-tok shape:** `can_run` returns `True`
  unconditionally and `replay` always calls the compiled `_compiled_fn`; no
  fallback/recompile logged in the measured window. ✓
- **Statistics:** full-reuse n=2 [257, 246], cold n=2 [747, 747] — tight. ✓ **Valid.**

## 4. Kernel microbenchmark (`kernels/bench/`)
- **Shapes/inputs:** matched Gemma-4 config (head_dim 256, 32 q / 16 kv heads,
  sliding_window 1024, bf16, batch 1), cold self-attention (prefix_len=0) — the
  fairest apples-to-apples unit. ✓
- **FLOP formula:** identical for both engines (dense S², not causal-discounted) →
  the *ratio* is layout-independent; TFLOP/s is a consistent dense-equivalent
  proxy, explicitly not an MFU claim. ✓
- **Timing:** ≥15 warmup, median of 60, `torch.cuda.Event`, kernel-call only. ✓
- **Layout-asymmetry caveat that *strengthens* the result:** vLLM reads K/V from a
  paged cache (block gather) while SGLang reads contiguous tensors — if anything
  this disadvantages vLLM, yet vLLM still wins the SWA shape ~8×. The asymmetry
  works *against* the conclusion, so it can't be the cause. ✓
- **Powerful external cross-check:** weighting the S=991 per-layer medians by the
  real 50:10 SWA:full layer mix predicts **721 ms** vs **724 ms** observed cold
  prefill — near-exact. A microbench timing the *wrong* computation would not
  reproduce the end-to-end number to 3 ms. Strong indirect validity. ✓
- **GAP FOUND → CLOSED.** The completed bench originally **timed** the kernels but
  never **asserted numerical correctness**. The Tier 1+2 run
  ([../kernels/bench/RESULTS_tier1_tier2.md](../kernels/bench/RESULTS_tier1_tier2.md),
  `bench_correctness.py`) added a correctness table: **every kernel — vLLM
  unified, SGLang stock, SGLang patched, FlashInfer, SDPA — passes vs an fp32-SDPA
  oracle** (max-abs ~0.008 ≪ 0.02 tol) at S∈{991,2048}, both full and SWA. This
  **retroactively validates the original RESULTS.md timings** (they were of correct
  computations — no timing invalidated). The patched kernel is bitwise identical to
  stock (max-abs = 0.0), so its 10.9× speedup is a pure valid win. **Gap closed —
  kernel result now fully valid, and independently corroborated by the 721≈724
  end-to-end prediction.**

## 5. Underlying baselines (corrected re-run, `rerun_clean.sh`)
Already audited when produced: the four defects D1–D4 closed
([rerun-defects.md](rerun-defects.md)), fairness verified against both engines'
source ([fairness-audit.md](fairness-audit.md)), n=3 with tight spread, all cells
`ttft_growth_ratio ≤ 1.04` (genuinely sub-knee). The decomposition's loaded
numbers come from here. ✓

## Cross-cutting confounds
- **Thermal / clock drift:** clocks are unlocked on the pod (uncontrolled). Not a
  threat to these results — the effects measured (8× SWA, +85% prefill) dwarf any
  drift, and n=2–3 spreads are tight (e.g. cold inductor [747, 747]). The main
  matrix mitigated it by engine-order alternation.
- **Warm-GPU / order bias:** each engine ran in its own venv/process sequentially;
  a first-vs-second warmup effect is small relative to an 8× gap. Not material.

## Verdict
**All experiments are valid.** The single gap — explicit numerical-correctness
verification of the kernel bench — is now **closed**: all kernels pass vs an
fp32-SDPA oracle, retroactively validating the original timings, and the patched
kernel is bitwise-identical to stock. The conclusion is independently corroborated
by the 721 ms ≈ 724 ms end-to-end prediction and by the patch (10.9× with
identical output confirms the diagnosed root cause).
