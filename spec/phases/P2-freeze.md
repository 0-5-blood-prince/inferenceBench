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

2-minute open-loop sweeps per workload (Full reuse, Cold, Partial reuse) on
**one** engine to locate the saturation knee `κ` — the rate where
completed/offered starts falling.

- Rates for the matrix: `{0.5κ, 0.8κ, 1.2κ}` per workload
  ([SPEC §6](../SPEC.md)).
- Cold's knee will be far below Full reuse's (≈10× the prefill work). Do not
  share a rate grid across workloads.
- Which engine hosts the pilot is recorded; the knee is used for rate
  *selection* only, never as a result.

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
