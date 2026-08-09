# P3 — Core matrix (3:15–6:00)

| | |
|---|---|
| **Objective** | All 28 core runs, each valid or explicitly tagged |
| **Entry** | P2 exit: frozen commit + shared rate grid |
| **Exit** | 28/28 runs with `completed ≥ 95%` or tagged `saturated`/`jit_contaminated`/`rows_exhausted`; metrics + GPU-clock snapshots present for every run |
| **Fallback** | If behind, drop the `1.2κ` overload row first — it is excluded from pooled analysis anyway. If still behind, the late-pass replicates are the second thing to trim, not the core 18 — see [SPEC §6](../SPEC.md) |

> **Amended before the matrix ran.** Run count moved from 22 to 28 and the
> variance design changed — see [SPEC §6](../SPEC.md)'s amendment note and
> [`../../learnings/measurement/p3-statistical-review.md`](../../learnings/measurement/p3-statistical-review.md)
> for the full review this responds to.

## The matrix

Core: 3 workloads × 2 engines × 3 rates = 18, plus Single stream × 2 = 20.

Plus the **late-pass variance block**: Full reuse @ `{0.8κ, 1.2κ}`, both
engines, each cell run 3× total (2 replicates beyond the original) = **8
runs**. These do not run back-to-back with the originals — see "Late pass"
below.

**28 runs total**, ≈150 min of runtime against this phase's 2h45 (165 min).
The remaining ~15 min is slack for the run that OOMs, not stretch budget —
tighter than before, because the 8 extra runs are what convert "one run beat
one run" into an actual variance estimate, which SPEC §6 now requires before
any hypothesis threshold is interpreted.

Contingency (+1 run, only if H4a is ambiguous): one Partial-reuse cell with
vLLM block size doubled.

VisionArena replay and Mooncake replay sit outside this matrix (2 + 8 runs;
[P6](P6-last-experiment.md)) and never displace core runs — a clean core matrix
is what makes the H7 prediction possible at all.

## Execution order

Engine order alternates per workload block; restart servers between workloads;
never rely on a runtime cache-clear. Engines are never co-resident (~60 GB
weights each). **Rate order within every block is fixed ascending**
(`0.5κ → 0.8κ → 1.2κ`), both engines, every workload — decided now, not
per-block on the day, so any thermal/allocator drift correlates with a fixed
rate-position rather than an implicit, unrecorded choice (SPEC §6).

```
Full reuse:     vLLM(0.5k, 0.8k, 1.2k) → SGLang(0.5k, 0.8k, 1.2k)
Cold:           SGLang(0.5k, 0.8k, 1.2k) → vLLM(0.5k, 0.8k, 1.2k)
Partial reuse:  vLLM(0.5k, 0.8k, 1.2k) → SGLang(0.5k, 0.8k, 1.2k)
Single stream:  SGLang(1)              → vLLM(1)
```

For Cold, the cache-disable flag is set at server launch for that block.

## Late pass — the variance block

Run **after** the block above, as its own pass — the whole point is that it
lands later in the ~150-minute session than the original Full reuse cell, so
it captures session drift rather than only short-term repeatability:

```
Full reuse late pass:  vLLM(0.8k ×2, 1.2k ×2) → SGLang(0.8k ×2, 1.2k ×2)
```

Each of the 4 cells (`{0.8κ, 1.2κ} × {vLLM, SGLang}`) gets 2 additional runs
here, joining its original from the block above for n=3 per cell. Same
per-run procedure as everywhere else — no shortcuts because it's "just" the
variance block; a replicate that skipped validity is not a replicate.

## Per-run procedure

Every path below is inside that workload's own folder — a run never writes
outside the workload it belongs to.

1. Scrape `/metrics` → `workloads/<name>/metrics/<run_id>_pre.txt`; scrape GPU
   clocks/temp/power → `workloads/<name>/metrics/<run_id>_gpu_pre.csv`
2. Run the harness cell (50-request warmup discarded; 200 requests or 4 min),
   writing `workloads/<name>/results/<run_id>.json`
3. Scrape `/metrics` → `workloads/<name>/metrics/<run_id>_post.txt`; scrape GPU
   clocks/temp/power → `workloads/<name>/metrics/<run_id>_gpu_post.csv`
4. Run `scripts/check_jit_contamination.py` against the engine log slice
   covering this run; it tags `jit_contaminated` in the result JSON directly
   if a JIT/capture event fired during measurement
5. **Validity, immediately, before the next run:**
   - `completed / offered ≥ 95%`, **or** `ttft_growth_ratio > 2.0` (second-half
     median TTFT over first-half median, in arrival order) — either tags
     `saturated`. Added after the real matrix run showed the gap directly:
     16 of the first 28 cells had TTFT growing to 10–180s while completion
     ratio stayed a perfect ~1.0, because the engine still finished every
     request, just far too slowly — exactly this section's own definition of
     saturation ("open-loop TTFT is a function of run length, the queue grows
     without bound"), which completion ratio alone cannot see. The threshold
     was verified against all 28 real cells before being fixed: every
     genuinely healthy one measured 1.00–1.07, every visibly collapsed one
     measured ≥2.4 — wide, unambiguous separation
     ([`learnings/measurement/ttft-growth-signal.md`](../../learnings/measurement/ttft-growth-signal.md))
   - cached-token fraction in band: Full reuse within ±5 pts of ≈85–90%, Cold
     ≈0, Partial reuse inside its constructed span
     ([SPEC §6](../SPEC.md)) — out of band means the workload is broken and the
     run is void, not just noisy
   - `jit_contaminated` present → exclude from pooled analysis, same as
     `saturated`
   - `rows_exhausted` present → the pre-built workload ran out of requests
     before `min_seconds` elapsed, so the run measured less time than intended
     ([`learnings/measurement/headroom-math.md`](../../learnings/measurement/headroom-math.md));
     exclude and rebuild that workload with a higher `--max-rate` before
     retrying the cell
6. `scripts/render_readme.py <name>` — regenerate that workload's README so its
   results table includes the run just finished
7. Append run_id, tags, and one-line status to the run log

Checking validity and re-rendering per-run rather than at 6:00 is the
difference between losing one run and losing a block. `run.sh cell` already
performs steps 1–4 and 6 automatically; step 5 and 7 are the operator's.

## Exit checklist

- [ ] 28 result JSONs + 56 metrics snapshots + 56 GPU-clock snapshots
      committed, each under its own workload folder
- [ ] Every run tagged: `ok` / `saturated` / `jit_contaminated` / `rows_exhausted` / `void(reason)`
- [ ] Every workload README regenerated and reflecting its final run set
- [ ] Late-pass variance block present: 8 runs, both engines, both rates
- [ ] Rate order within every block confirmed ascending in the run log
- [ ] If P3 exits by 5:30 → [P5](P5-stretch.md); else straight to
      [P4](P4-writeup.md)
