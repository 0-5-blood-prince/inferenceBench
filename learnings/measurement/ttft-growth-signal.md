# The completion-ratio validity gate missed half the real matrix

The real P3 matrix ran — 28 cells, all real GPU time, no shortcuts. Reviewing
the actual results found that **16 of 28 cells** had genuinely collapsed
under queueing (TTFT growing to 10–180 seconds) while tagged `ok`, because
`client.py`'s only saturation signal was `completion_ratio < 0.95`, and that
ratio stayed a perfect ~1.0 throughout — the engine finished every single
request, just far too slowly.

This is not a new phenomenon; it is the exact one this study's own spec
already names: *"open-loop TTFT is a function of run length — the queue
grows without bound"* (SPEC §6). It is also the same lesson
[pilot-methodology.md](pilot-methodology.md) already learned once, for the P2
pilot sweep specifically — completion ratio arrives too late, TTFT is the
earlier and truer signal. That fix was built into `pilot.py`. It was never
carried into `client.py`'s own tagging, which is what the real matrix
actually uses. The gap sat there through every review pass on the harness
mechanics, because none of those reviews were looking at *statistical*
correctness of the tag — they were checking that cells ran, wrote files, and
didn't crash. This one only became visible by reading the real numbers.

## The signal, and why it needed its own definition

`pilot.py` could compare each point against the *first* point in its sweep as
a baseline. A single P3 matrix cell has no such baseline — it's one rate, one
run. The fix operationalizes SPEC's own sentence directly instead: split a
run's successful requests into first-half and second-half by arrival order,
compare median TTFT of each half.

```python
def ttft_growth_ratio(ok):
    reqs = sorted(ok, key=lambda r: r["index"])
    n = len(reqs)
    first = [r["ttft_s"] for r in reqs[: n // 2]]
    second = [r["ttft_s"] for r in reqs[n // 2 :]]
    m1 = statistics.median(first)
    return statistics.median(second) / m1 if m1 > 0 else None
```

A flat, healthy run has ratio near 1.0. A queue growing without bound shows a
second half far slower than its first. Threshold (`> 2.0`) was chosen by
computing this ratio against all 28 real cells first, not picked in the
abstract:

| Class | Ratio range |
|---|---|
| Every genuinely healthy cell | 1.00 – 1.07 |
| Every visibly collapsed cell | 2.42 – 5.23 |

Wide, unambiguous separation — nothing sits near the cutoff. One cell
(`cold_vllm_0.8k`) was caught by the ratio (4.17) despite its overall p50
(1058ms) looking unremarkable on a quick visual scan — its p99 had already
reached 10s, and the growth ratio picked up the degrading second half that a
single overall percentile hid. The ratio is a better-calibrated signal than
eyeballing p50 against a threshold, not just a formalization of the same
thing.

## No cell was re-run

The raw per-request data was always valid — TTFT under real queueing is
genuine measured behaviour, not corruption. Only the *tag* was wrong.
`scripts/retag_ttft_growth.py` recomputes `ttft_growth_ratio` and re-derives
`tags` in place on the already-collected result JSONs, using the identical
function now built into `client.py`'s live `summarize()` so future runs (P5,
P6, any re-run) get it correctly the first time. Zero additional GPU cost.

## What this changes about reading the P3 results

Of the 16 newly-`saturated` cells, most are at `1.2κ` (already excluded from
pooled analysis by design — expected, not news) — but several are at `0.8κ`,
which was supposed to be one of the two "healthy, sub-saturation" comparison
points the core hypotheses (H1, H4b, etc.) actually depend on:

- **Cold**: both engines saturated by `0.8κ` — only `0.5κ` is clean for Cold.
- **Full reuse**: SGLang saturated at `0.8κ` (all three reps, including the
  late-pass replicates); vLLM stays clean through `0.8κ`, only saturating at
  `1.2κ`.
- **Partial reuse**: SGLang saturated at `0.8κ`; vLLM stays clean through
  `0.8κ`.

This is the resolution limit predicted in [pilot-results.md](pilot-results.md)
made concrete: the coarse geometric pilot sweep could only bracket the true
knee, not pinpoint it, and for SGLang specifically across two of three core
workloads, the true knee sits earlier in that bracket than the shared
`min(κ)` grid assumed. Full reuse and Partial reuse's `0.8κ` cell is a
genuine, symmetric, apples-to-apples comparison only on vLLM; on SGLang it is
already past-knee data, same interpretive caveat as `1.2κ`. Cold has no clean
above-baseline comparison point on either engine — only `0.5κ` is usable.
This must be stated plainly in the P4 writeup, not discovered again there.
