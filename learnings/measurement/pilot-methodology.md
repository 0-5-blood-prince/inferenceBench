# Pilot methodology: two stopping-condition bugs, a crash, and what actually detects a knee

Context: [what a saturation knee is and why each workload needs its own »](../../LEARNINGS.md).
This module covers `scripts/pilot.py`, the P2 tool that sweeps rate for one
workload and finds where it saturates.

## The stopping-condition trap in `client.py`

`client.py`'s run loop stops at `max(num_requests reached, min_seconds elapsed)`
— **not** `min`. The docstring even says so ("whichever is longer"), because
that's the right rule for the real matrix (SPEC: "200 requests or 4 min,
whichever is longer" guarantees both a sample-size floor and a wall-clock
floor). But it is the wrong rule to reuse naively for a pilot sweep, and it
produced two separate bugs before the actual mechanism was understood.

**Bug 1 — a fixed `num_requests=1000` "cap".** At a low pilot rate (0.5 req/s),
the loop will not even consult the time-based stop condition until 1000
requests have arrived — which takes 1000/0.5 = 2000 seconds, not the intended
120. A single pilot point ran for 33 minutes before being caught and killed.
The bug wasn't "duration didn't bound the run" — the loop design makes
`num_requests` an *unconditional floor*, so a too-high count means "wait
longer to reach the count," never "stop sooner."

**Bug 2 — the first fix, in the wrong direction.** The immediate instinct was
"give it headroom": `num_requests = rate * duration * 2.5`. This is *still* an
overestimate of the natural arrival count in `duration` seconds, so the same
failure mode recurred, just smaller — a point that should take 120s took over
300s. The fix had the right shape (compute `num_requests` from the target
rate) but the wrong sign: any margin *above* the expected arrival count just
means "the count is reached later than duration would have ended things
anyway," extending the run, not bounding it.

**The actual fix:** `num_requests` should be a *small* margin **above** the
Poisson mean (`rate * duration`), just enough that arrival-count variance
doesn't exhaust the pre-sliced request rows before `duration` elapses — around
1.1x, not 2.5x. And crucially: **warmup rows count too.** The loop iterates
over `warmup + num_requests` total rows, so real wall time is closer to
`(warmup + num_requests) / rate`, not `num_requests / rate` — a detail that
matters disproportionately at short pilot durations, where warmup is a larger
fraction of the budget.

**General lesson:** before trusting new orchestration code against a live GPU,
run it once at a cheap, short duration (a 20-second smoke test caught the
still-wrong-direction bug 2 before it repeated at full 120-second scale across
three workloads). This should have been standard practice from the first
attempt, not adopted only after a second failure.

## The crash that ate three workloads' worth of good data

With timing more or less fixed, all three pilot sweeps (Full reuse, Cold,
Partial reuse) ran, collected several genuinely useful points each — then
**crashed** with an unhandled `subprocess.TimeoutExpired` on a later point, and
took every already-collected point down with them. The bug: `pilot.py` only
wrote its output JSON once, at the very end of the per-workload loop. An
exception anywhere in that loop meant nothing was ever saved.

Fixed two ways, together:

- **Save after every point**, not just at the end. A crash now leaves whatever
  was already learned on disk.
- **Catch the per-point exception** (including the subprocess hard-timeout) and
  record it as a data point — `completion_ratio: null`, an error string — rather
  than letting it propagate and kill the sweep.

## The real finding: completion-ratio saturation detection fires too late

The crash wasn't random — it was caused by a deeper problem with the
saturation *definition* itself. SPEC's own stated rule is "the rate where
completed/offered starts falling." Using only that rule, the sweep pushed
rate higher and higher because completion ratio kept reading a perfect
**1.000** — while p50 TTFT was doing this, on every workload tested:

| Workload | Last point before blowup | Next point tried |
|---|---|---|
| Full reuse | 2.92 req/s, TTFT 260ms, ratio 1.000 | 5.25 req/s, TTFT **19.7s**, ratio 1.000 |
| Cold | 0.90 req/s, TTFT 555ms, ratio 1.000 | 1.62 req/s, TTFT **19.1s**, ratio 1.000 |
| Partial reuse | 1.62 req/s, TTFT 373ms, ratio 1.000 | 2.92 req/s, TTFT **31.4s**, ratio 1.000 |

A 40–100x TTFT explosion, and completion ratio never noticed. The engine was
still eventually finishing every request — just after a queueing delay that
had already gone unbounded. Pushing one rate step further into that regime is
what triggered the subprocess timeout above.

This is **not a deviation from SPEC** — it's a more faithful reading of a
sentence already in it: *"above the saturation knee, open-loop TTFT is a
function of run length (the queue grows without bound)"* (SPEC §6). Unbounded
TTFT growth under open-loop **is** SPEC's own definition of past-knee; it is
simply the earlier-arriving symptom, and completion ratio is the later,
blunter one.

**Fix:** `pilot.py` now stops a sweep on *either* signal — completion ratio
below threshold, **or** p50 TTFT exceeding 8x the first point's baseline. The
κ values this produces for the three core workloads (derived from the
already-collected bracket between last-healthy and first-collapsed points,
using the same interpolation logic the fixed script applies automatically):

| Workload | κ (req/s) |
|---|---|
| Full reuse | ≈ 4.09 |
| Cold | ≈ 1.26 |
| Partial reuse | ≈ 2.27 |

Cold's knee sitting well below the other two matches SPEC's own prediction
(~10x more prefill work per request than Full reuse).

Related: [the new per-point diagnostics (KV usage, preemptions, TPS) added
alongside this fix »](metrics-instrumentation.md)
