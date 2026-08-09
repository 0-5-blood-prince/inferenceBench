# P2 pilot results: κ per core workload, and two distinct saturation mechanisms

Produced by the fixed `scripts/pilot.py` (see
[pilot-methodology.md](pilot-methodology.md) for the bugs that had to be fixed
first) with the metrics instrumentation from
[metrics-instrumentation.md](metrics-instrumentation.md).

**Amended mid-P2:** originally piloted on vLLM only. Corrected to pilot both
engines and take `κ = min(κ_vllm, κ_sglang)` per workload — piloting one
engine and transferring its grid silently assumes the two engines saturate at
the same rate, which is exactly what a scheduler/batching comparison exists to
test, not presuppose. Full reasoning in
[p3-statistical-review.md](p3-statistical-review.md) and the amendment note in
[SPEC §6](../../spec/SPEC.md).

| Workload | κ_vllm | κ_sglang | Shared κ = min | Rate grid {0.5κ, 0.8κ, 1.2κ} |
|---|---|---|---|---|
| Full reuse | 4.082 | 4.082 | 4.082 | 2.041 / 3.266 / 4.899 |
| Cold | 1.26 | 1.26 | 1.26 | 0.630 / 1.008 / 1.512 |
| Partial reuse | 2.268 | 2.268 | 2.268 | 1.134 / 1.814 / 2.722 |

Every workload tied exactly — expected, not a coincidence to marvel at: see
"κ resolution is bounded by the sweep grid" below, which explains why a tie is
the typical outcome in this regime rather than evidence the engines perform
identically. Because every tie already equals `min()` of itself, **the rate
grid already written into `run.sh` from the vLLM-only pilots needed no
numeric change** — but the verification was still the right thing to do:
nothing before running SGLang's own sweep could have told us it would tie, and
in a different model/workload pairing it may not.

Ordering matches SPEC's own prediction (Cold ≈10x more prefill work per
request than Full reuse → lowest knee; Full reuse cheapest per request →
highest knee).

## Two different saturation mechanisms, visible only because preemption counts were added

**Full reuse and Partial reuse saturate via preemption thrashing** — vLLM
evicting in-flight requests to free KV capacity for others:

| Workload | Preemptions before collapse | Preemptions at collapse point |
|---|---|---|
| Full reuse | 0 (through 2.92 req/s) | **270** (at 5.25 req/s, TTFT 19.7s) |
| Partial reuse | 0 → 6 (at 1.62 req/s, TTFT still 378ms) | **84** (at 2.92 req/s, TTFT 30.9s) |

Partial reuse shows a visible early warning (6 preemptions one step before the
real collapse) that Full reuse does not — its preemption count goes straight
from 0 to 270 in one step. Worth remembering when reading the matrix's
preemption counters later: "zero preemptions last point" does not mean "one
rate step of headroom remains."

**Cold saturates with zero preemptions, at every point tested, including
collapse** (19.1s TTFT at 1.62 req/s). This is a genuinely different failure
mode: Cold has no shared image, so there is no KV to contend over in the way
Full reuse and Partial reuse do — nothing to evict. Its bottleneck is raw
prefill compute/bandwidth throughput, not cache/scheduling contention. This
matches the workload's own construction (SPEC §5: Cold does roughly 10x the
prefill work of Full reuse per request, with cache disabled by flag on top)
and is a mechanistic distinction worth carrying into the P4 writeup: a
TTFT-vs-rate curve alone would not tell you *why* three workloads saturate
differently, but the preemption counter does.

## κ resolution is bounded by the sweep grid, not by the engine

All three workloads' pilots (vLLM and SGLang, run separately per the
shared-grid amendment above) reported **identical** κ values. Not a physical
coincidence, and not specific to Full reuse: the sweep uses one fixed geometric ladder
(`0.5, 0.9, 1.62, 2.916, 5.249, ...`, same for every engine), and both
engines' completion ratio stayed a perfect 1.000 at *both* the last-healthy
and first-collapsed rungs — which is exactly the pilot-methodology finding
that TTFT blows up before completion ratio ever moves. With `c0 == c1 == 1.0`,
the interpolation formula's only fallback is a straight 50/50 midpoint between
the two bracketing rungs. That midpoint is a function of the *grid spacing*,
not of the engine's true knee — any two engines whose real knees fall
anywhere inside the same `[2.916, 5.249]` window report the same κ. The tool
cannot resolve finer than one rung in this regime.

What the same two points *do* still show, uncontaminated by that limitation,
is **severity at the shared rate** — read directly off the raw TTFT, not
through the interpolation. This held across all three workloads, same
direction every time, SGLang worse in every case:

| Workload | Rate | vLLM TTFT | SGLang TTFT | Ratio |
|---|---|---|---|---|
| Full reuse | 5.249 | 19.4s | 44.1s | 2.3x |
| Cold | 1.620 | 19.1s | 51.5s | 2.7x |
| Partial reuse | 2.916 | 30.9s | 56.9s | 1.8x |

Partial reuse additionally shows the gap opening *before* the shared collapse
point: at rate 1.62 (still completion-ratio-healthy on both), vLLM read 378ms
and SGLang read 2.14s — already 5.7x, a full rung before either engine
"collapsed" by the coarse definition.

Same coarse collapse threshold every time, sharply different blowup once past
it, and a consistent direction across all three workload shapes. That is
arguably more informative than the κ numbers themselves, and it survives the
resolution limit because it's a direct reading, not an interpolation. See
[../infrastructure/cuda-graphs.md](../infrastructure/cuda-graphs.md) for the
leading explanation: SGLang disables CUDA-graph capture for prefill on
multimodal models entirely (its own log names the incompatibility), while
vLLM graphs 51 prefill-covering buckets — checked and ruled out as an
explanation: the attention-*kernel* backend (Triton) is identical on both
engines, confirmed from both engines' own logs, so this is specifically about
graph capture, not kernel choice.

This does not block using `κ = min(κ_vllm, κ_sglang)` for the shared rate
grid — both engines resolving to the same bracket means the grid is valid
regardless of which one is nominally lower — but it does mean κ should never
be read as "the two engines have proven-identical saturation points." A
tighter sweep (smaller `--factor`) would narrow the bracket at the cost of
more pilot points; not done here since SPEC already treats κ as
rate-*selection*-only, and the grid's validity doesn't depend on the exact
value within the bracket.

## A gauge-sampling limitation worth knowing about

The `num_requests_running`/`num_requests_waiting` gauges (added mid-sweep,
present on Partial reuse's points but not the earlier two) read **0** even at
the collapse points — including 30.9s TTFT. This is not a contradiction of the
preemption evidence; it's a sampling artifact. The post-point scrape happens
*after* the client subprocess returns, by which time the request backlog that
caused the collapse has already fully drained (the harness waits for every
in-flight request before exiting). So point-in-time gauges caught here
describe the state *after* the storm, not *during* it — only the diffed
*counters* (preemptions, cache hits) correctly show what accumulated across
the point's full duration. If peak concurrency during a run is ever needed
directly (rather than inferred from preemption/TTFT), it would require
sampling the gauge periodically *during* the run, not just pre/post.
