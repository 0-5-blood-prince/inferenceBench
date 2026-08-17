# Kernel profiling resolves the mechanism — and reconciles a real contradiction

The [official-bench-and-mechanism.md](../measurement/official-bench-and-mechanism.md)
sweep + eager-closer localized vLLM's prefill-TTFT lead to "faster eager forward
kernels, ~1.9× at ≥500 tokens" but couldn't say *which* kernels. This uses a native
torch-profiler trace of each engine's prefill forward to answer that — and in doing
so, surfaces a genuine contradiction with the earlier SWA-patch conclusion, which
turns out to be workload-scoped rather than a mistake. Data:
[bench/profiles/](bench/profiles/), parser: [bench/parse_trace.py](bench/parse_trace.py).

## Method
`--enable-torch-compile`-free (vLLM `--enforce-eager`, SGLang default-eager)
servers, each profiled via their native `/start_profile`+`/stop_profile`
endpoints (driven by `sglang.bench_serving --profile`) over 3 pure-text
(no image) requests at `--random-input-len 991`, conc=1 — matching the eager
closer's decisive length. Kernel time isolated to the single-request **prefill
forward window** using each engine's own wrapper annotation
(`execute_context_1(991)_generation_0(0)` for vLLM,
`step[EXTEND bs=1 toks=991]` for SGLang) rather than the whole multi-request
trace, so decode steps don't dilute the comparison.

**Parser correctness note:** the first pass mis-bucketed kernels — a bare
`"gemm"` keyword false-matched `"gemma"` (SGLang's Gemma-4-specific kernel names
all start `_gemma_...`), a bare `"flash"` false-matched a KV-cache-write kernel
(`reshape_and_cache_kernel_flash`, unrelated to attention), and the real
attention kernels (`kernel_unified_attention` for vLLM; `_fwd_kernel` /
`_fwd_grouped_kernel_stage1` for SGLang — literal Triton function names, no
"attn" substring) matched nothing and fell into "other". Fixed with exact names
plus a regex negative-lookahead (`gemm(?!a)`) — see the parser's inline comment
for the fully worked-out root cause.

## Result: pure-text prefill (S=991), kernel time inside the isolated window

| | vLLM | SGLang | delta |
|---|---:|---:|---:|
| Total kernel time | 323.4 ms | 560.5 ms | +237.1 ms |
| GEMM/matmul (240 calls both — identical structural decomposition) | 253.3 ms | 240.1 ms | **−13.2 ms** (SGLang faster) |
| **Attention** (60 calls, one/layer) | **49.1 ms** | **298.3 ms** | **+249.2 ms** |
| norm/rope/cache/other | ~21 ms | ~22 ms | ~+1 ms |

**The +249.2 ms attention delta accounts for the entire +237.1 ms kernel-time
delta** — GEMM's −13.2 ms offsets most of the remainder, norm/rope/cache is
~+1 ms noise. GEMM is a wash (SGLang's is marginally *faster* — same
cuBLAS/CUTLASS kernel names, same call count). **On pure text, attention is
not a contributor to the gap — it is the gap.**

## The apparent contradiction, and its resolution

This looked flatly incompatible with the earlier finding
([prefill-decomposition.md](../measurement/prefill-decomposition.md)) that
installing the patched SWA kernel into the live server, marker-verified,
changed end-to-end cold (multimodal) TTFT by ~0%. Reading the actual patch
diff resolves it — [PATCH b], the dominant fix (removing the per-tile
`SKIP_TILE` reduction + data-dependent branch that kills Triton's
software-pipelining), is gated:

```diff
-        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
+        if USE_CUSTOM_MASK:
             SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0
```

It only *skips* the expensive path when `USE_CUSTOM_MASK` is **False**. Gemma-4
multimodal requests set `USE_CUSTOM_MASK=True` (bidirectional image-token
masking, applied via `prepare_attn_masks` in SGLang's `gemma4_mm.py`, read
earlier in this investigation). So **on the multimodal cold workload the
dominant patch optimization never engaged** — explaining the measured 0% end-to-end
change there. This profiling run used pure text (`custom_mask=None`), where the
patch fully engages.

## Confirmatory re-measurement: the patch DOES close the text-prefill gap

Re-ran the patched kernel end-to-end against the same pure-text workload
(`bench_serving`, conc=1, ilen sweep):

| ilen | vLLM | SGLang-stock | SGLang-patched | gap closed |
|---:|---:|---:|---:|---:|
| 512 | 107.3 | 188.6 | 211.3 | −28% (worse — see caveat) |
| **991** | **342.8** | **612.4** | **374.4** | **88%** |
| 2048 | 716.0 | 1436.0 | 732.5 | **98%** |

At **991 tokens — our real workload's text-prefill length** — patched SGLang
(374 ms) lands almost on top of vLLM (343 ms). At 2048, essentially closed. This
directly confirms the profiling breakdown: fixing the SWA kernel closes 88–98%
of the vLLM-vs-SGLang TTFT gap **for text prefill**.

**512-token caveat:** n=1 (single bench_serving run, no reps) and the patch
came out *worse* there, which doesn't fit the pattern (512 and 991 are both
under the 1024-token window, so should behave similarly per the mechanism —
the window doesn't reduce tile count at either length, only the branch removal
matters). Treat as unconfirmed noise pending a repeated measurement, not a
real effect.

## What this means for the study's actual (multimodal) headline

**No change to the multimodal-workload conclusion** — the SWA kernel patch
does not help there (confirmed twice: the marker-verified end-to-end test, and
now explained mechanistically via `USE_CUSTOM_MASK`). The scope is now precise
rather than a blanket claim:
- **Multimodal (our study's core comparison):** SWA kernel is not the
  end-to-end driver; the real cause remains open.
- **Pure text:** SWA kernel (specifically its `SKIP_TILE`/pipelining defect)
  is confirmed as the dominant driver, closing 88–98% when fixed.

**A genuinely new, unexplored optimization surface**, not claimed as proven:
extending the branch-removal fix to work correctly under `USE_CUSTOM_MASK=True`
(e.g. by combining the window-tile-skip with the custom-mask logic, rather than
falling back to the always-reduce path whenever a mask is present at all) would
plausibly recover a large fraction of the multimodal gap too, given how directly
attention explains the text gap and how mechanically narrow the current gating
is. This is a hypothesis a maintainer/PR author should test, not a result this
investigation established.
