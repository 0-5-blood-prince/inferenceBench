# inferenceBench — project resume points

Benchmarking **vLLM 0.26.0** vs **SGLang 0.5.16** under multimodal prefix caching,
single A100 80GB, model `google/gemma-4-31B-it`. Overarching question: *does
prefix reuse change which serving engine wins?* Pre-registered in `spec/`, results
and mechanism in `learnings/`. Two resume points below: the **benchmark study**
(what wins) and the **mechanism investigation** (why).

---

## Resume Point 1 — The benchmark study (P0 → P4)
*Commits `76b7e1e` … `430dda4`. Runbook: `spec/phases/P0`–`P4`. Data: `workloads/*/results/`.*

### What was done
- **Pre-registration discipline** established: `spec/` is a frozen runbook; every
  change is a new commit with a stated-reason blockquote, corrections registered
  *before* the run that acts on them.
- **P0–P2**: feasibility, gates, workload freeze. Dropped Gate A as a blocker (it
  tested a property a latency comparison can't see — [learnings/gate-a/](learnings/gate-a/));
  pinned the output budget; fixed cache-metric misreads.
- **P3 core matrix** (28 cells): rate sweep × 3 workloads (full-reuse,
  partial-reuse, cold) × 2 engines. Critical retag mid-flight — completion-ratio
  saturation detection fired 40–100× too late; replaced with a **TTFT-growth
  signal** ([ttft-growth-signal.md](learnings/measurement/ttft-growth-signal.md)).
- **P4 text-only control**: image stripped, replaced 1:1 by pad tokens — the gap
  shrinks ~⅓ but survives on pure text.
- **Fairness audit** ([fairness-audit.md](learnings/measurement/fairness-audit.md)):
  verdict fair-with-two-disclosed-asymmetries; KV pools within ~13%
  (vLLM ~18.1k tok, SGLang ~16.2k), both FCFS, both chunked prefill.
- **Four defects (D1–D4)** found auditing the finished matrix (no restart between
  rate cells so Cold wasn't cold; knee miscalibrated; CUDA-graph confound; one-sided
  JIT contamination) — all closed by one pre-registered **corrected re-run**
  ([rerun-defects.md](learnings/measurement/rerun-defects.md)).

### Headline result ([rerun-results.md](learnings/measurement/rerun-results.md))
**vLLM leads TTFT by 43–71% at clean load on every workload; p99 ≈ 2–2.5× SGLang.**

| Workload | vLLM p50 | SGLang p50 | gap |
|---|---|---|---|
| Full reuse 0.5κ | 205 ms | 311 ms | +52% |
| Partial reuse 0.5κ | 314 ms | 534 ms | +70% |
| Cold (cache off) 0.66κ | 567 ms | 812 ms | +43% |

Prefix reuse does **not** flip the winner; it narrows nothing decisive. SGLang
saturates at ~45–55% of vLLM's load. ITL (decode) is identical → the whole
disadvantage is in the prefill/TTFT path.

### To resume P1
The P4 writeup (`spec/phases/P4-writeup.md`) builds on the corrected data; figures
in `figures/`. Optional future work (deferred): branching/multi-turn reuse,
Mooncake cache-tier.

---

## Resume Point 2 — The mechanism investigation (why SGLang's prefill is slower)
*Commits `a01047d` … `f6854a8`. Writeup: [prefill-decomposition.md](learnings/measurement/prefill-decomposition.md). Data: `decomp/`. Kernels: [learnings/kernels/](learnings/kernels/).*

A chain of experiments, each ruling out a hypothesis, converging on the root cause.

1. **Decomposition** (`decomp.sh`): at concurrency=1 (queue≈0), SGLang's *prefill
   compute alone* is **+85%** full-reuse (132→244 ms) / **+49%** cold (485→724 ms),
   while the **queue is symmetric** (≤6 ms between engines). → Not scheduler/admission.
2. **Not caching** (Cold cache-off still +43–49%), **not decode** (ITL identical).
3. **Config sweep** (`config_test.sh`, `prefill_graph_test.sh`): torch.compile 0%
   (decode-only), graph-of-eager +3–5%, FlashInfer/FA3 **rejected** (Gemma-4 pins
   Triton). All null.
4. **Correction — vLLM is NOT FlashAttention.** Startup logs: *both* engines run
   `TRITON_ATTN` on A100 (FA skipped by the bidirectional-vision requirement,
   architectural not hardware — an H100 would not unlock FA for Gemma-4 on vLLM).
5. **The real untested lever** (`inductor_prefill_test.sh`): prior "prefill graph"
   tests used the default `tc_compiler=eager`. The genuine vLLM analog is
   `--cuda-graph-tc-compiler inductor` (Inductor-compiled prefill). Measured:
   full-reuse **252 (+3%)**, cold **747 (+3%)** — Inductor genuinely engaged
   (414 s compile), still **null**.
6. **Root cause (final verdict).** The attention *kernel* is walled off from
   compilation in both engines (`torch.compiler.disable` in SGLang, `splitting_ops`
   in vLLM), so no compilation lever can touch it. "Both Triton" ≠ same kernel:
   they are two independent implementations — SGLang's `extend_attention_fwd`
   (general materialized `custom_mask` path) vs vLLM's `unified_attention`
   (in-kernel causal+SWA). **The gap is the Triton kernel implementation itself**,
   ~1.5–1.9×, unreachable by any SGLang flag; the faster backends are forbidden
   for Gemma-4 on A100. Published data agrees (FA2-CUDA ~1.3–1.5× > Triton-FA on
   A100; SGLang docs recommend FlashInfer over Triton on Ampere).

### Ruled out, with data
| Hypothesis | Verdict |
|---|---|
| Prefix caching | No — cold cache-off +43–49% |
| Decode / kernel efficiency | No — ITL identical |
| Scheduler / queue | No — queue symmetric ≤6 ms |
| Compilation / CUDA graphs | No — compile 0%, graph +3–5%, **Inductor +3%** |
| Different attention *algorithm* | No — both Triton |
| **Triton kernel *implementation*** | **Yes — the root cause** |

### In flight / pending
- **Kernel microbenchmark** (subagent, running on the pod): head-to-head timing of
  the two Triton kernels at Gemma-4 shapes → `learnings/kernels/bench/`
  (`bench_vllm.py`, `bench_sglang.py`, `RESULTS.md`). Will give the direct
  kernel-level ratio to confirm the ~1.5–1.9× end-to-end gap.
- **Open option (not started):** rent an **H100** to test SGLang `trtllm_mha`
  (Hopper-only fused kernel, in Gemma-4's accepted list) vs its Triton — the one
  hardware experiment that could show a faster kernel closing the gap. Needs cost
  approval before renting.

### To resume P2
Read [prefill-decomposition.md](learnings/measurement/prefill-decomposition.md)
(the full chain + final verdict) and [learnings/kernels/](learnings/kernels/) (the
two kernels + bench). All `decomp/*.json` are raw per-request data.

---

## Live state (as of this resume)
- **Pod** `ktblk948n3l6dc` (213.173.105.5:30032) — up, GPU idle, running only the
  kernel-bench subagent. **Should be stopped once the bench finishes** to halt billing.
- **Uncommitted:** `decomp/cold_sglang_indprefill_conc1_rep2.json` (pulled local);
  will commit with the bench artifacts.
- **Everything else committed** through `f6854a8`.
