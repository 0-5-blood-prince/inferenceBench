# P3 statistical review: five findings, all verified, before any real cell ran

An external methodology review of the P3 design, done before the core matrix
started (P2's pilots and smoke run were complete; no real matrix cell had yet
run). The headline: operationally the design was strong — pre-registration
placement, per-run validity discipline, restart-over-cache-clear, the shared
`min(κ)` grid ([pilot-results.md](pilot-results.md)) — but statistically thin
exactly on the thing the study exists to produce: engine-to-engine deltas.

Each claim was checked against real evidence on this pod before acting on it,
not accepted on the strength of the argument alone.

## 1. n=1 per cell, and the one variance probe was at the wrong operating point

Every one of the 18 core cells was a single run. The only variance estimate
was Full reuse @ 0.8κ, duplicated once — n=2, one degree of freedom, not a
real standard deviation. Worse: latency variance fans out as ρ→1 (the
queueing-delay formula from the knee-definition reading list — wait time
∝ ρ/(1−ρ) — makes this exact), so a duplicate measured at the low-variance
0.8κ point systematically **underestimates** variance at 1.2κ, which is
exactly where an interval would matter most.

**Fix:** replicate `{0.8κ, 1.2κ}` for **both** engines, each cell to n=3 (2
degrees of freedom) — 8 additional runs. Run them as a **separate late pass**,
after the rest of the matrix, not back-to-back with the originals:
back-to-back duplicates measure short-term repeatability, which is the wrong
thing to measure for a matrix spanning ~150 minutes. See [SPEC §6](../../spec/SPEC.md)
and [P3-core-matrix.md](../../spec/phases/P3-core-matrix.md) for the resulting
design.

## 2. 20-request warmup was too short — verified, not assumed

Checked directly: `grep -rln "jit_monitor" logs/*.log` on the pod matched
**ten separate log files**, including
`"Triton kernel JIT compilation during inference: kernel_unified_attention.
This causes a latency spike; consider extending warmup to cover this
shape/config"` — vLLM's own engine, in its own words, reporting a JIT
compilation event firing *during* a measured window, not during warmup. Twenty
requests does not reliably touch every batch-size bucket or tensor shape that
triggers a distinct kernel compile or graph capture (see
[../infrastructure/cuda-graphs.md](../infrastructure/cuda-graphs.md) for why
graphs are bucketed at all). Any cell that happened to hit one of these got a
latency spike injected into its TTFT/ITL numbers, invisible to every existing
validity check.

**Fix, two parts:**
- Warmup raised 20 → 50 requests (`scripts/client.py`), to make hitting an
  untouched bucket during measurement less likely.
- That alone cannot guarantee coverage, so a **detector**, not just a bigger
  warmup: `scripts/check_jit_contamination.py` scans the engine log slice
  produced during each cell's own measurement window (line-count delta from
  immediately before the cell started) for the JIT-warning pattern, and tags
  the run `jit_contaminated` in its own result JSON if found — wired into
  `run.sh cell()` automatically. Before this, a JIT-contaminated run silently
  passed as `ok`; now it's excluded from pooled analysis the same way
  `saturated` already is.

## 3. "200 requests or 4 min" gives inconsistent, tail-starved samples — and a real survivorship bias

Verified in `scripts/client.py`: `ok = [r for r in results if r["error"] is
None and r["ttft_s"] is not None]`, and every percentile (`ttft_s`, `itl_s`,
`isl_tokens`, `osl_tokens`) is computed **only over `ok`**. At or near
saturation, the requests most likely to fail or time out are exactly the
slowest ones — excluding them from the percentile calculation makes the
reported tail look better than it is. Combined with the OR-based stop
condition, sample count varies wildly by cell (roughly 200 at a fast rate,
~120–240 at Cold's low 0.5κ under the 4-minute cap) — a p99 from 200 samples
is effectively the second-worst sample, not a stable estimate.

**Fix:** SPEC §6 now requires ≥500 completed requests before a cell's p99 is
reportable as anything other than descriptive; below that, label it as such.
This compounds with finding 4 below for why 1.2κ specifically never gets a
point estimate.

## 4. 1.2κ is a transient snapshot, not a steady state

Already true in the original design (1.2κ was already excluded from pooled
analysis), but the *reason* wasn't spelled out and the door was open to
someone quoting a 1.2κ TTFT number anyway. Near ρ→1 the queue does not
converge inside a finite measurement window — SPEC §6 already says this in
different words ("open-loop TTFT is a function of run length, the queue grows
without bound"). A 1.2κ latency number is therefore a property of how long
you happened to run the cell, not of the engine.

**Fix:** SPEC §6 now states explicitly that 1.2κ supports only
ordinal/qualitative claims ("engine A degrades more gracefully than B at
overload"), never a quoted point estimate, in the P4 writeup or anywhere else.

## 5. Thermal/clock drift was uncontrolled *and unlogged*, and rate order within a block was unspecified

Checked `spec/phases/P3-core-matrix.md`'s original execution-order text: it
specifies which **engine** goes first per workload block (counterbalanced,
confirmed present), but says nothing about the order of the three **rates**
within a block. Combined with GPU clocks being impossible to lock from inside
this container ([../infrastructure/environment.md](../infrastructure/environment.md)
— a fact already known before this review, from P0), a ~150-minute session
with unrecorded rate ordering means any drift that exists cannot be
distinguished from a rate effect after the fact.

**Fix, both cheap:**
- Rate order fixed **ascending** (`0.5κ → 0.8κ → 1.2κ`) in every block, for
  every workload, both engines — decided in the spec now, not chosen ad hoc
  on the day.
- GPU clocks/temperature/power now scraped alongside every existing
  `/metrics` pre/post scrape (`nvidia-smi --query-gpu=clocks.sm,clocks.mem,
  temperature.gpu,power.draw`), wired into `run.sh cell()`. Free — the
  scraping infrastructure already existed for `/metrics`; this is one more
  `nvidia-smi` call alongside it.

## Net effect on the matrix

Run count: 22 → **30** (8 additional late-pass replicates). Time budget:
~110 min → **~150 min**, against a 165-minute phase window — tighter, but the
8 extra runs are what make the engine-to-engine deltas defensible rather than
"one run beat one run" dressed up with a pre-registration commit. Everything
else in the original design (per-run validity, restart discipline, engine-order
counterbalancing, cached-fraction band checks, the `min(κ)` shared grid) was
correct as written and is unchanged.
