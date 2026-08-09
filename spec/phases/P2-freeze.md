# P2 — Freeze (2:30–3:15)

| | |
|---|---|
| **Objective** | Design frozen in git; rates fixed from pilots; harness smoke-tested |
| **Entry** | P1 exit: gates passed |
| **Exit** | The pre-registration commit exists; `κ` per workload measured; one clean end-to-end smoke run |
| **Fallback** | If pilots run long, trim to 2 rates per workload — decided *before* the commit, not after |

## The commit

Commit the entire `spec/` directory — this is the pre-registration act. It sits
here deliberately: **after** gates and pilots (late enough to be informed),
**before** any measured engine comparison (early enough to be honest).

After this commit:
- Hypotheses H1–H7 and all thresholds in [SPEC §3](../SPEC.md) are frozen.
- Workload constants from P0 (`T`, canonical block length `C`, `E_min`,
  `E_max`, resolution, image counts) are recorded in the commit.
- Any later change to `spec/` is a new commit with a stated reason — never a
  silent amend.

## Pilots — find each workload's knee

> **Amended before any matrix run — pilot BOTH engines, not one.** Originally
> written as a single-engine sweep with the choice of engine merely "recorded."
> That silently assumes the two engines saturate at the same rate, which is
> exactly the kind of thing a scheduler/batching comparison is supposed to
> test, not presuppose. If the engines' knees differ and only one is piloted,
> the resulting grid can put one engine's "healthy, sub-saturation" cell at or
> past the *other* engine's knee — at which point every delta in that cell is
> confounded by load, not attributable to the engine. The fix costs a few
> extra 2-minute pilot points (~15-20 min against a 3-hour matrix) and removes
> a real risk, so it is corrected here rather than carried into P3.

2-minute open-loop sweeps per workload (Full reuse, Cold, Partial reuse), on
**both** engines, to locate each one's saturation knee — the rate where
completed/offered starts falling, or (see [SPEC §6](../SPEC.md) and
[`../../learnings/measurement/pilot-methodology.md`](../../learnings/measurement/pilot-methodology.md))
where p50 TTFT blows past its baseline, whichever comes first in practice.

- **The rate grid used in the matrix must be identical for both engines.**
  Testing each engine at its own knee confounds every delta with a load
  difference — you cannot tell "faster" from "tested gentler." One shared
  grid, applied to both.
- Per workload: `κ = min(κ_vllm, κ_sglang)`. This keeps `0.5κ` and `0.8κ`
  genuinely sub-saturation for **both** engines — the comparison that matters
  most — and makes `1.2κ` a real past-knee probe for at least the weaker
  engine. If the two knees are far apart, the `1.2κ` cell will floor the
  weaker engine while the stronger one copes; that asymmetry is itself
  interpretable, not a defect.
- Rates for the matrix: `{0.5κ, 0.8κ, 1.2κ}` per workload, from that shared κ.
- Cold's knee will be far below Full reuse's (≈10× the prefill work). Do not
  share a rate grid across *workloads* — the grid is shared across engines
  only, within one workload.
- **The per-engine knee divergence is itself a signal, not just a
  calibration input.** Same model, same GPU, same KV budget — if the two
  knees differ substantially, record why (scheduler policy, KV capacity,
  preemption behaviour). κ stays rate-*selection*-only and is never a
  headline H-series result, but the gap between `κ_vllm` and `κ_sglang` is
  worth a line in the P4 writeup regardless.

## Smoke run

One full cell end to end — server up, warmup, 200 requests, metrics scraped
pre/post, JSON written, server down — to prove `run.sh` mechanics before the
matrix. A smoke failure here costs 10 minutes; the same failure at run 14 of 22
costs the afternoon.

## Exit checklist

- [ ] `spec/` committed; commit hash noted in the run log
- [ ] `κ` for Full reuse, Cold, and Partial reuse recorded; rate grid written
      into `run.sh`
- [ ] Smoke run artifacts present (results JSON + two metrics snapshots)
- [ ] Single stream needs no pilot (concurrency 1) — confirmed configured
