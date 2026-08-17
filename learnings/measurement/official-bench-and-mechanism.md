# Official-tool validation + the resolved prefill mechanism

Two things: (1) validating our custom `client.py` results against SGLang's own
`sglang.bench_serving`, and (2) using that tool's input-length sweep to finally
pin *why* vLLM's prefill TTFT beats SGLang's — after the SWA-kernel hypothesis was
refuted end-to-end ([kernels/why-sglang-swa-slow.md](../kernels/why-sglang-swa-slow.md),
[prefill-decomposition.md](prefill-decomposition.md)). Data:
[../kernels/bench/official/](../kernels/bench/official/).

## Harness comparison — ours vs `sglang.bench_serving`

Read both in full (`sglang/benchmark/serving.py`). **Equivalent**: TTFT = first
streamed token − send; ITL = inter-token gaps excluding TTFT; output length
hard-pinned (`min_length`/`min_tokens`); Poisson arrivals; conc=1 = serialized.
**Differences that matter**:
- **Endpoint asymmetry (theirs):** bench_serving sends SGLang `/generate` and vLLM
  `/v1/completions` — different code paths. Ours hits identical `/v1/chat/completions`
  on both. **Checked and ruled out:** SGLang via `sglang-oai` (`/v1/completions`,
  same as vLLM) = 616 ms, identical to `/generate` 616 ms.
- **Workload:** their `random` is text-only, unique prompts — can't reproduce our
  multimodal/caching finding (needs `image` + `generated-shared-prefix`).
- **Warmup:** theirs defaults to 1 request; ours 10–50 (the JIT-contamination guard).

## Validation: the headline reproduces on the official tool

`bench_serving`, random text, 991-token input, conc=1, A100:

| | vLLM | SGLang | gap |
|---|---:|---:|---:|
| Median TTFT | 347 ms | 616 ms | **+78%** |
| Median ITL | 43.5 ms | 43.2 ms | ~0% |

Independently confirms both core findings: **vLLM leads TTFT** (in our 43–85% range)
and **ITL is identical** (decode equal). On pure text (no image, no cache) — so the
gap is not vision, not caching, not decode.

## Mechanism: input-length sweep + eager closer

Median TTFT (ms), conc=1, same `/v1/completions` endpoint for both:

| input tok | vLLM-compiled | vLLM-eager | SGLang | SGLang/vLLM |
|---:|---:|---:|---:|---:|
| 128 | 64 | 107 | 108 | 1.69× |
| 256 | 65 | 111 | 120 | 1.83× |
| 512 | 107 | 121 | 189 | 1.76× |
| 991 | 343 | 359 | 612 | 1.79× |
| 2048 | 716 | 744 | 1436 | 2.01× |

**Finding 1 — not fixed framework overhead.** The gap scales with input length (44 ms
at 128 → 720 ms at 2048); at short inputs both engines are cheap and close. SGLang
runs a fairly constant ~1.8× slower (a *multiplicative* per-token cost), so it's the
prefill forward, not a per-request IPC/scheduling tax.

**Finding 2 — not compilation, at the lengths that matter.** Forcing vLLM to
`--enforce-eager` (no Inductor, no CUDA graph — verified in the log): at long prefill
vLLM-eager ≈ vLLM-compiled (991: 1.05×, 2048: 1.04×), **yet vLLM-eager is still
~1.9× faster than SGLang** (991: 1.71×, 2048: 1.93×). With *both engines eager*,
vLLM still wins ~1.9×. So compilation/CUDA-graph is **not** why vLLM leads at
prefill lengths ≥ ~500 tokens.

**Finding 3 — compilation IS a short-sequence win.** At 128 tokens vLLM-eager (107)
≈ SGLang (108), while vLLM-*compiled* is 64 — so at short prefill the entire gap is
vLLM's CUDA-graph eliminating fixed launch overhead (which dominates a tiny forward).
SGLang can't use it (prefill graph auto-disabled for this multimodal model).

## Resolved mechanism
The vLLM prefill-TTFT lead is **two effects**:
1. **Short prefill → CUDA-graph/compilation** (vLLM has it, SGLang doesn't): a
   fixed-launch-overhead win, ~1.7× at 128 tokens.
2. **Long prefill → faster eager prefill-forward kernels** (~1.9×, *not* compilation):
   the dominant driver at our ~991-token workload. Both engines eager, vLLM still ~1.9×.

> **CORRECTION (superseded by op-level profiling): the ≥500-tok gap is
> attention-dominated, NOT MLP-dominated.** This section originally guessed the
> ~1.9× was the "whole eager forward kernel set (MLP-dominated)" with attention a
> small share — that was written before any op-level profile existed. A subsequent
> torch-profiler trace of each engine's isolated prefill-forward window (pure text,
> S=991) shows the opposite: **GEMM/MLP is a wash** (SGLang's is even marginally
> faster, same kernel names, same call count) and the **entire ~237 ms kernel-time
> gap is attention** (vLLM `kernel_unified_attention` 49 ms vs SGLang
> `_fwd_kernel`+`_fwd_grouped_kernel_stage1` 298 ms, a ~249 ms delta). So the SWA
> kernel finding is *not* separate from the eager-forward gap — on pure text it **is**
> the gap, and the patch closes it 88–98% end-to-end. Full breakdown:
> [../kernels/profiling-resolves-mechanism.md](../kernels/profiling-resolves-mechanism.md).
> (Scope: this is the pure-text mechanism; on the multimodal workload the SWA patch
> is null via `USE_CUSTOM_MASK` gating and the driver stays open — see that file.)
