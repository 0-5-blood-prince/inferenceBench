# vLLM vs SGLang under multimodal prefix caching: a benchmark and a root-cause investigation

*Technical writeup. For the scannable summary see [`README.md`](README.md); for
the narrative version see [`BLOG.md`](BLOG.md); for topic-by-topic detail see the
modules indexed in [`LEARNINGS.md`](LEARNINGS.md).*

## Abstract

We ask whether prefix-cache reuse changes which of two production LLM serving
engines — vLLM 0.26.0 and SGLang 0.5.16 — delivers lower latency, serving a
31B multimodal model (`google/gemma-4-31B-it`) on a single A100 80GB. The study
was pre-registered before any measurement. The headline: **prefix reuse does not
change the winner.** vLLM leads p50 time-to-first-token (TTFT) by 43–71% at every
clean comparison point, with a p99 tail 2–2.5× SGLang's; decode (inter-token
latency) is identical. Reuse only modulates the *size* of the gap. The second
half of the report is a mechanism investigation that localizes the gap to the
prefill forward pass, rules out caching, decode, scheduling, compilation, and
attention *algorithm*, and — for pure-text prefill — down to the sliding-window
attention kernel, while leaving the *multimodal* end-to-end root cause explicitly
open. Two leading hypotheses were disproven with data along the way; both
reversals are reported.

## 1. Setup

| | |
|---|---|
| Model | `google/gemma-4-31B-it`, bf16, `max_model_len=8192` |
| Hardware | A100 80GB PCIe (primary); H100 NVL for cross-hardware spot-checks. RunPod. |
| Engines | `vllm==0.26.0`, `sglang==0.5.16` — never co-resident (two ~60 GB weight sets don't fit 80 GB) |
| Input | ~1010 tokens/request, identical across workloads; fixed 448×448 image → 258 tokens (Gemma-4 is not dynamic-resolution) |
| Output | `max_tokens=128, ignore_eos=true, min_tokens=budget` (hard-pinned so both engines do equal work) |
| Arrivals | open-loop Poisson; concurrency-1 closed loop for the decomposition |
| Workloads | full-reuse (87.5% cacheable), partial-reuse (30–80%), cold (caching off); text-only controls |

Both engines were run under matched, defaults-at-equal-memory conditions. A
source-level fairness audit confirmed measured KV pools within ~13% (vLLM ~18.1k,
SGLang ~16.2k tokens), both FCFS, both chunked prefill, and symmetric shared-image
encoder caching ([fairness-audit.md](learnings/measurement/fairness-audit.md)).

## 2. Methodology and its corrections

**Pre-registration.** The question, hypotheses (H1–H7), workload geometry, and
measurement thresholds were frozen in [`spec/SPEC.md`](spec/SPEC.md) at the P2
commit. Every subsequent change is a dated, stated-reason commit in the phase
runbook it affects — corrections are visible, not silent.

Three methodology results are worth stating before the numbers, because each one
changed a number:

1. **Gate A dropped.** The original design gated on byte-identical output between
   reused and fresh prefill. It tests a property no latency comparison needs and
   manufactured a false engine asymmetry out of bf16 reduction-order noise.
   Replaced by Gate R (reuse-by-counters + budget pinning + shape matching).
   [learnings/gate-a/](learnings/gate-a/).
2. **Saturation was invisible to the original signal.** 16 of 28 core cells were
   in queueing collapse (TTFT growing without bound within the run) while
   reporting a *perfect* completion ratio. A `ttft_growth_ratio` signal (second-
   half/first-half median TTFT, threshold 2.0) caught them; they were excluded,
   not re-run. [ttft-growth-signal.md](learnings/measurement/ttft-growth-signal.md).
3. **Four defects, one re-run.** Auditing the finished matrix found: cold cells
   not actually cold above 0.5κ (servers not restarted between rate cells, so
   vLLM's multimodal processor cache stayed warm), a miscalibrated saturation
   knee, an un-measured CUDA-graph confound, and one-sided JIT contamination on
   vLLM's clean points. All four were closed by a single pre-registered re-run.
   [rerun-defects.md](learnings/measurement/rerun-defects.md).

The numbers below are from that corrected re-run (n=3, restart before every cell).

## 3. Results

p50 TTFT, corrected re-run, at clean sub-saturation load:

| Workload | rate | vLLM | SGLang | gap | p99 (vLLM/SGLang) |
|---|---|---|---|---|---|
| Full reuse | 0.5κ | 205 ms | 311 ms | +52% | 312 / 631 |
| Full reuse | 0.65κ | 216 ms | 336 ms | +56% | 337 / 730 |
| Partial reuse | 0.5κ | 314 ms | 534 ms | +70% | 634 / 1353 |
| Partial reuse | 0.75κ | 323 ms | 552 ms | +71% | 680 / 1652 |
| Cold (cache off) | 0.66κ | 567 ms | 812 ms | +43% | 922 / 1641 (p90) |

- **Inter-token latency is identical**: full reuse 51.3 vs 50.4 ms, cold 47.1 vs
  47.3, partial 48.5 vs 48.7. Per-token decode compute is equal; the disadvantage
  is entirely in the prefill/TTFT path.
- **Reuse modulates the gap but never flips or closes it.** The gap shrinks as
  cached fraction rises; even at 87.5% reuse vLLM still leads. Cache *matching
  granularity* (vLLM 16-token blocks vs SGLang single-token radix) is ruled out
  quantitatively: measured cached fractions differ by only ~1.3 pts.
- **SGLang saturates earlier**, at ~45–55% of vLLM's sustainable load. For these
  short-output workloads throughput is prefill-bound, so a slower prefill path
  caps throughput sooner.

**Pre-registered hypotheses.** H1 (SGLang wins under full reuse) — *falsified,
sign reversed*. H2 (engines within 10% cold) — *falsified*. H3 (throughput
ranking prefix-invariant) — *confirmed, ordinal*. H4a (cached fraction differs
<5 pts) — *confirmed, 1.3–1.4 pts*. H4b (partial>full gap only at high rate) —
*falsified, present at low rate*. H5 (ITL within 10%) — *confirmed, ~0%*. H6/H7
not run.

## 4. Mechanism investigation

### 4.1 Decomposition — it's prefill compute, not queue

At concurrency 1, one request is in flight, so TTFT ≈ prefill compute with no
queue. SGLang's prefill alone is **+85%** (full reuse: 132→244 ms) / **+49%**
(cold: 485→724 ms), while the queue component (loaded TTFT − conc-1 TTFT) is
**symmetric** between engines (≤6 ms). This rules out cache-aware scheduling /
admission overhead as the driver. Combined with identical ITL and a cold
(caching-off) gap of +43–49%, three hypotheses die here: **not caching, not
decode, not scheduling.** [prefill-decomposition.md](learnings/measurement/prefill-decomposition.md).

### 4.2 First wrong turn: "it's FlashAttention vs Triton" — false

The natural assumption was that vLLM uses FlashAttention and SGLang uses Triton.
The engine startup logs refute it: **both** select a Triton attention backend on
A100 (`AttentionBackendEnum.TRITON_ATTN`). FlashAttention is rejected for this
model (head_dim 256 + the bidirectional-vision requirement), and the fused
`trtllm_mha` backend — the one config-available alternative — requires **Blackwell
(SM100)**, rejected on both A100 and H100. So the comparison is Triton-vs-Triton;
"different attention algorithm" is not the cause.

### 4.3 Config levers — all null

torch.compile changes prefill 0% (it targets decode); forcing the prefill CUDA
graph on moves TTFT ±5%; even a genuine Inductor-compiled prefill
(`--cuda-graph-tc-compiler inductor`, verified engaged) lands at +3%. No
compilation/graph lever closes the gap.

### 4.4 Second wrong turn: the kernel patch that didn't matter

A head-to-head **kernel microbenchmark** showed SGLang's sliding-window (SWA)
Triton attention kernel is ~8× slower than vLLM's *in isolation* on A100 (an
Ampere-specific effect: on H100 the same kernel is only ~1.3× its own causal
path). Source reading found why: SGLang applies the window as a per-tile mask
plus a data-dependent `SKIP_TILE` reduction/branch that defeats Triton's
software-pipelining, and never narrows the loop to the window. A minimal patch
(narrow the loop bound + drop the redundant branch) made the kernel **10.9×
faster with bitwise-identical output**.

**Then the patch changed multimodal end-to-end TTFT by 0%** (721 vs 724 ms cold,
marker-verified the live server loaded the patched module, caches purged). A
stock re-check reproduced the baseline, ruling out harness drift. Conclusion at
this stage: **the SWA kernel is not the end-to-end bottleneck for the multimodal
workload** — a direct refutation of the kernel-microbench framing.

### 4.5 Ruling out compilation, then profiling

An **input-length sweep** (128→2048 tokens, both engines, concurrency 1) plus a
vLLM `--enforce-eager` control settled the compilation question: at prefill
lengths ≥ ~500 tokens, vLLM-eager ≈ vLLM-compiled, **yet vLLM-eager is still
~1.9× faster than SGLang** — so compilation is not the driver at the lengths that
matter (it is a fixed-overhead win only at very short prefill). The gap is a
per-token prefill cost.

**Op-level GPU profiling** (torch profiler, isolated to the single-request
prefill-forward window, pure text, S=991) locates it:

| Category | vLLM | SGLang |
|---|---|---|
| GEMM / matmul (240 calls each) | 253 ms | 240 ms (a wash — SGLang marginally faster) |
| **Attention** (60 calls, one/layer) | **49 ms** | **298 ms** |
| norm / rope / cache / other | ~21 ms | ~22 ms |
| **Total kernel time in window** | **323 ms** | **560 ms** |

The +249 ms attention delta ≈ the entire +237 ms window delta. On pure text,
attention is not a *contributor* to the gap — it *is* the gap.

### 4.6 Reconciliation

The Step-4.4 contradiction (patch = 10.9× kernel, 0% multimodal) resolves in the
patch diff: its dominant fix (dropping the `SKIP_TILE` reduction/branch) is gated
`if USE_CUSTOM_MASK:` — it only engages when custom masking is **off**. Gemma-4
multimodal requests set `USE_CUSTOM_MASK=True` (bidirectional image-token
masking), so the fix never fired on the multimodal workload. Pure-text requests
have it off. A confirmatory re-run with the patch on the **pure-text** workload
closes **88% of the gap at 991 tokens and 98% at 2048** — exactly matching the
profiling breakdown. [profiling-resolves-mechanism.md](learnings/kernels/profiling-resolves-mechanism.md).

### 4.7 Where it lands

- **Pure text:** the vLLM-vs-SGLang prefill-TTFT gap is the sliding-window
  attention kernel; a minimal, output-preserving fix closes 88–98% of it.
- **Multimodal (the study's actual workload):** the SWA kernel is *not* the
  end-to-end driver (the fix is gated off by custom masking, and confirmed to
  move it 0%). The multimodal root cause **remains open**. Extending the fix to
  the custom-mask path is a plausible future direction, not a result established
  here.

## 5. Cross-validation

The headline was reproduced with SGLang's own tool, `sglang.bench_serving`, on a
matched text prompt at concurrency 1: **vLLM 347 ms vs SGLang 616 ms (+78%) TTFT,
ITL identical (43.5 vs 43.2 ms)**. A harness comparison confirmed the two
measure TTFT/ITL and pin output length equivalently; notably, `bench_serving`
sends SGLang and vLLM *different* endpoints (a confound our harness avoids), which
we ruled out by re-running SGLang on the shared endpoint (identical 616 ms).
[official-bench-and-mechanism.md](learnings/measurement/official-bench-and-mechanism.md).

## 6. Threats to validity

Single hardware generation and one model; synthetic noise images (encoder cost is
content-independent but real traffic has correlated structure this only
approximates); open-loop Poisson arrivals; scheduler defaults recorded, not
swept; GPU clocks not lockable from the container (thermal drift mitigated by
engine-order alternation and repeated variance blocks, not eliminated). The
mechanism section's kernel-level attribution is airtight for text but, by
construction, does not resolve the multimodal case. A full validity audit of the
mechanism chain is in [validity-audit.md](learnings/measurement/validity-audit.md).

## 7. Conclusions

Prefix reuse does not change which engine wins for this model on this hardware:
vLLM leads TTFT 43–71% at clean load across full-reuse, partial-reuse, and cold,
with identical decode. The advantage is entirely in the prefill forward pass, is
independent of caching/scheduling/compilation and of the attention *algorithm*
(both Triton), and — for text — is the sliding-window attention kernel
implementation specifically. The multimodal case, which is what the study set out
to measure, has a confirmed *location* (prefill forward) but an unresolved
*cause*. The most transferable output may be the methodology: pre-registration,
a defect-fixing re-run, source-level fairness auditing, independent-tool
cross-validation, and a mechanism hunt disciplined enough to disprove its own
leading hypotheses twice.
