# Does prefix reuse change which LLM serving engine wins?

**vLLM 0.26.0 vs SGLang 0.5.16 under multimodal prefix caching.** One model
(`google/gemma-4-31B-it`), A100 80GB (+ H100 spot-checks), pre-registered before
the first measured run — then a multi-week root-cause investigation into *why*
the answer came out the way it did.

The design, hypotheses, and thresholds were frozen in [`spec/SPEC.md`](spec/SPEC.md)
before any measurement; every later change is a dated, stated-reason commit. What
makes this repo unusual isn't the benchmark — it's the audit trail: four
measurement defects caught and corrected via a pre-registered re-run, a
source-level fairness audit, cross-validation against each engine's *own*
benchmark tool, and a mechanism hunt in which the leading hypothesis was
disproven twice before the evidence settled.

> **Full technical account:** [`WRITEUP.md`](WRITEUP.md) · **narrative version:**
> [`BLOG.md`](BLOG.md) · **topical deep-dives:** [`LEARNINGS.md`](LEARNINGS.md)

---

## The answer, in one paragraph

**No — prefix reuse does not change which engine wins.** vLLM had lower p50
time-to-first-token (TTFT) at every clean, sub-saturation comparison point, by
**43–71%**, with a tail (p99) roughly 2–2.5× SGLang's. The pre-registered
expectation was the opposite (SGLang's token-granular RadixAttention was
predicted to beat vLLM's 16-token block matching under full reuse); it did not,
at any reuse level. What reuse *changes* is the size of the gap, which shrinks as
the cached fraction rises but never closes. **Decode is identical** — inter-token
latency matches within ~1 ms engine-to-engine — so the entire disadvantage lives
in the prefill / first-token path. SGLang also saturates earlier, at roughly
**45–55% of vLLM's sustainable load**.

## Results (corrected re-run, n=3, p50 TTFT)

| Workload | rate | vLLM | SGLang | gap | tail (p99/p90) |
|---|---|---|---|---|---|
| Full reuse | 0.5κ | 205 ms | 311 ms | **+52%** | 312 / 631 |
| Full reuse | 0.65κ | 216 ms | 336 ms | **+56%** | 337 / 730 |
| Partial reuse | 0.5κ | 314 ms | 534 ms | **+70%** | 634 / 1353 |
| Partial reuse | 0.75κ | 323 ms | 552 ms | **+71%** | 680 / 1652 |
| Cold (cache off) | 0.66κ | 567 ms | 812 ms | **+43%** | 922 / 1641 (p90) |

These supersede the original single-shot matrix: an audit found four defects
(cold cells not actually cold above 0.5κ, a miscalibrated saturation knee, an
un-measured CUDA-graph confound, and one-sided JIT contamination), all closed by
one pre-registered re-run. See [`learnings/measurement/rerun-defects.md`](learnings/measurement/rerun-defects.md).

**Independently reproduced.** SGLang's own `sglang.bench_serving` tool, on a
matched text prompt at concurrency 1, gives **vLLM 347 ms vs SGLang 616 ms
(+78%) TTFT with identical ITL (43.5 vs 43.2 ms)** — so the finding is not an
artifact of our custom harness.

---

## Why vLLM wins — a mechanism hunt with two self-corrections

The interesting part is *why*, and the honest version includes the wrong turns.

**Step 1 — localize it.** At concurrency 1 (no queue), TTFT ≈ prefill compute.
SGLang's prefill is **+85%** (full reuse: 132→244 ms) / **+49%** (cold:
485→724 ms), while the queue component is symmetric between engines (≤6 ms).
Ruled out with data: **caching** (cold, caching off on both, still +43–49%),
**decode** (ITL identical), **scheduling** (queue symmetric). The gap is prefill
compute.

**Step 2 — a wrong turn, corrected.** "vLLM must be using FlashAttention, SGLang
Triton." **False** — the engine logs show *both* run a Triton attention backend
on A100 (`TRITON_ATTN`); FlashAttention is rejected for this model
(head_dim 256 + bidirectional vision masking), and the fused `trtllm_mha` backend
requires Blackwell (SM100) — unavailable on A100 *and* H100. So it's Triton vs
Triton, not a different algorithm.

**Step 3 — config levers, all null.** torch.compile (0%), forcing the prefill
CUDA graph on (±5%), even genuine Inductor-compiled prefill (+3%) — none close
the gap.

**Step 4 — the second wrong turn, corrected.** A kernel microbenchmark showed
SGLang's sliding-window attention kernel ~8× slower than vLLM's *in isolation*; a
minimal, bitwise-identical patch made it 10.9× faster. **But installed into the
live server it changed multimodal end-to-end TTFT by 0%** (marker-verified the
server loaded it). The SWA kernel is *not* the end-to-end bottleneck for the
multimodal workload.

**Step 5 — profile it.** An input-length sweep + a vLLM `--enforce-eager` control
proved it isn't compilation either (vLLM-eager still ~1.9× faster than SGLang).
Op-level GPU kernel profiling of the isolated prefill forward, on pure text,
shows: **GEMM/matmul is a wash** (SGLang's is even marginally faster, same
kernels, same call count), and **attention accounts for essentially the entire
gap** (vLLM 49 ms vs SGLang 298 ms across the 60 layers).

**The resolution — and its honest scope.** Reading the patch diff explained the
Step-4 contradiction: its dominant fix is gated `if USE_CUSTOM_MASK:` and only
engages when custom masking is *off*. Gemma-4 multimodal requests turn it *on*
(bidirectional image masking), so the fix never fired there. On **pure text**
(custom mask off) it fully engages — and a confirmatory re-run closes **88% of
the gap at 991 tokens, 98% at 2048**. Net:

- **Pure text:** the gap is the sliding-window attention kernel (confirmed).
- **Multimodal (the study's actual workload):** the SWA kernel is *not* the
  driver; the end-to-end root cause **remains open**. Stated as open, not papered
  over.

Full detail: [`learnings/kernels/profiling-resolves-mechanism.md`](learnings/kernels/profiling-resolves-mechanism.md)
and [`learnings/measurement/official-bench-and-mechanism.md`](learnings/measurement/official-bench-and-mechanism.md).

---

## How the finding was made trustworthy

- **Pre-registration.** Question, hypotheses, and thresholds frozen in `spec/`
  before measurement; corrections are dated commits, never silent edits.
- **Defect audit + re-run.** Four defects found in the finished matrix, fixed by
  one pre-registered re-run ([rerun-defects.md](learnings/measurement/rerun-defects.md)).
- **Fairness audit against source.** Measured KV pools within ~13% (vLLM ~18.1k,
  SGLang ~16.2k tokens), both FCFS, both chunked prefill, encoder caching
  symmetric ([fairness-audit.md](learnings/measurement/fairness-audit.md)).
- **Saturation discipline.** 16 of the original 28 cells were in queueing
  collapse while reporting a perfect completion ratio; caught by a TTFT-growth
  signal and excluded ([ttft-growth-signal.md](learnings/measurement/ttft-growth-signal.md)).
- **Independent tool cross-check.** `sglang.bench_serving` reproduces the headline.
- **Validity audit** of the whole mechanism chain ([validity-audit.md](learnings/measurement/validity-audit.md)).

---

## Reproducing

```bash
./run.sh bootstrap                     # venvs, weights, workload build (re-run after any pod resume)
./run.sh up vllm                       # start an engine
./run.sh cell full-reuse vllm 0.5k     # one cell
./run.sh p3                            # the core matrix
python scripts/analyze_rerun.py        # corrected-run tables from raw per-request JSON
python scripts/make_figures.py         # figures/
```

Note: RunPod stop/resume wipes the container disk (venvs, apt packages); the
network volume (weights) survives. `ninja-build` must be re-apt-installed — it's
an undeclared dependency of both engines.

## Repo map

| Path | What's in it |
|---|---|
| [`spec/SPEC.md`](spec/SPEC.md) | Frozen pre-registration: question, H1–H7, workloads, measurement contract |
| [`spec/phases/`](spec/phases/) | P0–P6 runbooks; post-freeze amendments as stated-reason blockquotes |
| [`workloads/`](workloads/) | One folder per experiment: requests, raw results, metrics scrapes |
| [`learnings/`](LEARNINGS.md) | ~18 topical modules on what broke and what it taught — indexed in `LEARNINGS.md` |
| [`learnings/kernels/`](learnings/kernels/) | The kernel investigation: microbench, profiling, source analysis, traces |
| [`WRITEUP.md`](WRITEUP.md) · [`BLOG.md`](BLOG.md) | Full technical account · narrative version |
| `run.sh` · `scripts/` | Orchestration; `client.py` load generator, analysis chain |

## Status & limitations

Single hardware generation, one model; synthetic noise images; open-loop Poisson
arrivals; engine scheduler defaults recorded, not swept; GPU clocks not lockable
from the container (thermal drift mitigated by engine-order alternation, not
eliminated). The multimodal end-to-end root cause is **open**. The kernel-patch
work is a validated *diagnosis* on text, not a shipped fix — no upstream PR is
claimed here.
