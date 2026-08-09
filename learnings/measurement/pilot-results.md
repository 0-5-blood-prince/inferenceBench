# P2 pilot results: κ per core workload, and two distinct saturation mechanisms

Produced by the fixed `scripts/pilot.py` (see
[pilot-methodology.md](pilot-methodology.md) for the bugs that had to be fixed
first) with the metrics instrumentation from
[metrics-instrumentation.md](metrics-instrumentation.md), on vLLM (the
recorded pilot engine, per SPEC's "which engine hosts the pilot is recorded").

| Workload | κ (req/s) | Rate grid {0.5κ, 0.8κ, 1.2κ} |
|---|---|---|
| Full reuse | 4.082 | 2.041 / 3.266 / 4.899 |
| Cold | 1.26 | 0.630 / 1.008 / 1.512 |
| Partial reuse | 2.268 | 1.134 / 1.814 / 2.722 |

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
