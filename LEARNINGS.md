# Learnings

Operational knowledge gathered while running this study. Not part of the frozen
pre-registration in `spec/` — this is what we found out by doing it, including
where the runbook itself was wrong.

This is organized by **topic**, not by phase — unlike `spec/phases/`, which is
a linear runbook you work through in order, the modules below are independent
and can be read in whichever order matches what you're debugging.

## [gate-a/](learnings/gate-a/) — Gate A: why it was dropped as a pre-registration blocker

The headline finding of the day. Gate A required reused KV to produce
byte-identical output to fresh prefill. It tested a property a latency
comparison structurally cannot see, and its short-window form manufactured a
false engine asymmetry.

- [mechanism.md](learnings/gate-a/mechanism.md) — the bf16 reduction-order cause,
  worked numerics, our measurements, and why the divergence event turned out
  not to be reproducible run-to-run
- [vs-latency.md](learnings/gate-a/vs-latency.md) — why no latency claim needs
  this gate at all, and what replaced it (Gate R)
- [deterministic-mode.md](learnings/gate-a/deterministic-mode.md) — why the
  engine-native fix (batch-invariant kernels) isn't available on the pinned
  SGLang version

## [measurement/](learnings/measurement/) — getting the numbers right

- [output-budget.md](learnings/measurement/output-budget.md) — pinning
  `max_tokens`/`ignore_eos`, and a client-retokenization measurement error
  that produced a false "vLLM ignores ignore_eos" claim
- [cache-metrics.md](learnings/measurement/cache-metrics.md) — why
  `sglang:cache_hit_rate` is a gauge that lies post-run, and the per-request
  signals that actually work on each engine
- [pilot-methodology.md](learnings/measurement/pilot-methodology.md) — two
  stopping-condition bugs in the P2 rate-sweep tool, a crash that nearly lost
  all three workloads' pilot data, and the real finding underneath it all:
  completion-ratio saturation detection fires 40-100x TTFT too late
- [pilot-results.md](learnings/measurement/pilot-results.md) — the resulting κ
  per core workload, and two genuinely different saturation mechanisms
  (preemption thrashing vs raw throughput limits) visible only because
  preemption counters were added
- [headroom-math.md](learnings/measurement/headroom-math.md) — a harness bug
  found by a broader code review after P2 closed: `run.sh cell()` never
  scaled request-row count to the cell's actual rate, so higher-rate P3 cells
  would have silently measured a third of their intended duration and still
  reported `ok`. The fix, why its margin (1.15x vs `pilot.py`'s 1.1x) is
  deliberately approximate rather than tuned, and the `rows_exhausted`
  assertion that catches it directly if the margin is ever insufficient
- [ttft-growth-signal.md](learnings/measurement/ttft-growth-signal.md) — the
  real P3 matrix ran (28 cells), and reviewing the actual results found 16 of
  them had genuinely collapsed under queueing (TTFT to 10-180s) while tagged
  `ok`, because completion ratio — the only saturation signal `client.py`'s
  live tagging used — stayed near a perfect 1.0 throughout. Fixed by adding
  a TTFT-growth signal (verified against all 28 real cells first) and
  retagging the already-collected data in place; no cell needed re-running.
  Also states plainly what this changes about reading the results: several
  `0.8κ` cells, meant to be a clean sub-saturation comparison point, are
  already past-knee on SGLang for two of three core workloads
- [p3-statistical-review.md](learnings/measurement/p3-statistical-review.md) —
  an external methodology review of the P3 design, done before any real cell
  ran: n=1 per cell with the wrong variance probe, a verified (not assumed)
  warmup-too-short bug, tail-latency survivorship bias, why 1.2κ can never be
  a point estimate, and unlogged thermal drift — all five claims checked
  against real evidence before acting on them
- [fairness-audit.md](learnings/measurement/fairness-audit.md) — post-hoc
  audit of whether the two engines got a fair comparison, checked against
  both engines' source at the pinned tags: verdict fair-with-two-disclosed-
  asymmetries, measured KV pools within ~13%, shared-image encoder caching
  symmetric on both engines, and a newly surfaced third mechanism candidate
  (vLLM's decode-prioritizing scheduler admitting nearly-free cached
  prefills vs SGLang's retract-and-re-prefill loop)
- [metrics-instrumentation.md](learnings/measurement/metrics-instrumentation.md) —
  closing the gap against NVIDIA's standard LLM benchmarking metric set
  (token-counted TPS, TPS-per-user, RPS, ISL/OSL, image/text token split,
  KV-cache utilization)

## [infrastructure/](learnings/infrastructure/) — the pod and the engines

- [cuda-graphs.md](learnings/infrastructure/cuda-graphs.md) — how CUDA graph
  capture works, and a structural, unequalizable asymmetry between the two
  engines' default capture behaviour that must be cited before attributing any
  TTFT gap to caching alone
- [environment.md](learnings/infrastructure/environment.md) — Runpod
  network-storage performance, CUDA forward-compatibility, and other pod-level
  facts that shaped `run.sh`
