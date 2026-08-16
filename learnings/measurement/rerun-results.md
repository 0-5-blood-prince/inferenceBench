# The corrected re-run: results and mechanism attribution

The pre-registered re-run (D1–D4 fixes, [rerun-defects.md](rerun-defects.md))
ran clean end to end: re-pilot → 30-cell clean matrix → graph A/B → graphs-on
knee pilot. Every clean cell is `ok`, every `ttft_growth_ratio ≤ 1.04` (all
genuinely sub-knee, not near-collapse), n=3 with tight spread, and p99 (cache
workloads) / p90 (cold) are reportable — the things D4 said were impossible at
the old n=1 clean point. Raw data in `workloads/*/results/`,
`workloads/*/pilot_*.json`, `gates/sglang-graph-ab/`.

## The clean comparison (p50 TTFT, n=3, SGLang vs vLLM)

| Workload | rate | vLLM p50 | SGLang p50 | gap | vLLM p99/p90 | SGLang p99/p90 |
|---|---|---|---|---|---|---|
| Full reuse | 0.5κ (0.94) | 205 ms | 311 ms | **+52%** | 312 | 631 |
| Full reuse | 0.65κ (1.50) | 216 ms | 336 ms | **+56%** | 337 | 730 |
| Partial reuse | 0.5κ (0.60) | 314 ms | 534 ms | **+70%** | 634 | 1353 |
| Partial reuse | 0.75κ (0.90) | 323 ms | 552 ms | **+71%** | 680 | 1652 |
| Cold (cache off) | 0.66κ (0.48) | 567 ms | 812 ms | **+43%** | 922 (p90) | 1641 (p90) |

The headline survives the corrected methodology and sharpens: **vLLM leads TTFT
by 43–71% at genuinely clean load on every workload, and the gap widens at the
tail** (SGLang p99 ≈ 2–2.5× vLLM).

## What the gap is NOT — four hypotheses ruled out with data

1. **Not prefix caching.** Cold runs with caching disabled on both engines
   (verified: `--no-enable-prefix-caching` + `--mm-processor-cache-gb 0` on
   vLLM, `--disable-radix-cache` + `SGLANG_VLM_CACHE_SIZE_MB=0` on SGLang) and
   SGLang is still **+43%** slower. Most of the gap has nothing to do with the
   feature the study set out to probe.
2. **Not decode / kernel efficiency.** Decode inter-token latency (ITL p50) is
   **identical** between engines on every workload — full reuse 51.3 vs 50.4 ms,
   cold 47.1 vs 47.3, partial reuse 48.5 vs 48.7. Same model, same Triton
   kernels; per-token decode compute is equal. **The entire disadvantage is in
   the prefill / time-to-first-token path.**
3. **Not the prefill CUDA-graph disable.** SGLang auto-disables prefill graph
   capture for this multimodal model; forcing it on
   (`--cuda-graph-backend-prefill tc_piecewise`, capture succeeded, no fault)
   changes sub-knee TTFT by **−1.6% / −1.4%** (full/partial) and the saturation
   knee by **0% / −1%**. The graph asymmetry accounts for ~1–2% of a 50–70%
   gap, from both the TTFT and the knee angle.
4. **Only ~⅓ is multimodal-specific.** The text-only control (earlier) shrank
   the gap by roughly a third but did not erase it — so the vision-input path
   contributes, but the majority of the gap is present on pure text too.

## What the gap IS — localized, not yet split

The whole effect lives in the prefill/TTFT path (ITL rules out decode), it is
not caching, not the graph disable, and mostly not multimodal. The saturation
knee tells the same story: SGLang saturates at **~45–55% of vLLM's load**
(full reuse κ 1.88 vs 4.07; cold 0.75 vs 1.30; partial reuse 1.21 vs 1.14 —
the one workload where they converge), and forcing graphs on does not move it.
For these short-output workloads throughput is prefill-bound, so a slower
prefill path caps throughput earlier — consistent with a lower knee.

Remaining live candidates, all on the prefill path, in rough order of support:

- **Scheduler / admission overhead.** SGLang's cache-aware radix-tree
  bookkeeping and longest-prefix-match sorting run on the critical path to the
  first token, every step, even when reuse is low. vLLM's FCFS admission does
  less critical-path work. Fits both the TTFT gap and the ~½ knee.
- **Gemma-4 hybrid-SWA prefill compute.** The SGLang log flags this model as a
  "Hybrid swa model"; the two engines may implement the sliding-window prefill
  attention with different efficiency (prefill only — decode/ITL is unaffected).
- **Multimodal input preprocessing (CPU).** Image hashing, embedding-cache
  lookup, radix `pad_value` insertion — the ~⅓ that the text-only control
  attributes to the multimodal path, separate from the (cached) ViT compute.

Splitting scheduler-overhead from prefill-compute needs a prefill-time-vs-
queue-time decomposition (SGLang server-side timing metrics, or an isolated
single-request prefill microbenchmark) — the clean next step, out of scope for
this run.

## Fairness note

All measured under matched, defaults-at-equal-memory conditions
([fairness-audit.md](fairness-audit.md)): measured KV pools within ~13%
(SGLang max_total_num_tokens ~16.2k, vLLM ~18k), both FCFS, both chunked
prefill, shared-image encoder cached on both engines, identical `requests.jsonl`
per workload. The two disclosed asymmetries (prefill-graph coverage, vLLM's
A100 batch-size carve-out) are now both shown not to drive the result — the
graph one directly (this file), the batch-size one bounded by vLLM winning
*despite* its smaller A100 default batch budget.
