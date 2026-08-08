# P3 — Core matrix (3:15–6:00)

| | |
|---|---|
| **Objective** | All 22 core runs, each valid or explicitly tagged |
| **Entry** | P2 exit: frozen commit + rate grid |
| **Exit** | 22/22 runs with `completed ≥ 95%` or tagged `saturated`; metrics snapshots present for every run |
| **Fallback** | If behind, drop the `1.2κ` overload row first — it is excluded from pooled analysis anyway |

## The matrix

Core: 3 workloads × 2 engines × 3 rates = 18, plus Single stream × 2, plus the
variance duplicate (Full reuse @ 0.8κ, both engines) × 2 = **22 runs** ≈ 110 min
of runtime against this phase's 2h45. The rest is slack for the run that OOMs,
not stretch budget.

Contingency (+1 run, only if H4a is ambiguous): one Partial-reuse cell with
vLLM block size doubled.

VisionArena replay and Mooncake replay sit outside this matrix (2 + 8 runs;
[P6](P6-last-experiment.md)) and never displace core runs — a clean core matrix
is what makes the H7 prediction possible at all.

## Execution order

Engine order alternates per workload block; restart servers between workloads;
never rely on a runtime cache-clear. Engines are never co-resident (~60 GB
weights each).

```
Full reuse:     vLLM(3 rates + dup) → SGLang(3 rates + dup)
Cold:           SGLang(3)           → vLLM(3)
Partial reuse:  vLLM(3)             → SGLang(3)
Single stream:  SGLang(1)           → vLLM(1)
```

For Cold, the cache-disable flag is set at server launch for that block.

## Per-run procedure

Every path below is inside that workload's own folder — a run never writes
outside the workload it belongs to.

1. Scrape `/metrics` → `workloads/<name>/metrics/<run_id>_pre.txt`
2. Run the harness cell (warmup 20 requests discarded; 200 requests or 4 min),
   writing `workloads/<name>/results/<run_id>.json`
3. Scrape `/metrics` → `workloads/<name>/metrics/<run_id>_post.txt`
4. **Validity, immediately, before the next run:**
   - `completed / offered ≥ 95%`, else tag `saturated`
   - cached-token fraction in band: Full reuse within ±5 pts of ≈85–90%, Cold
     ≈0, Partial reuse inside its constructed span
     ([SPEC §6](../SPEC.md)) — out of band means the workload is broken and the
     run is void, not just noisy
5. `scripts/render_readme.py <name>` — regenerate that workload's README so its
   results table includes the run just finished
6. Append run_id, tags, and one-line status to the run log

Checking validity and re-rendering per-run rather than at 6:00 is the
difference between losing one run and losing a block.

## Exit checklist

- [ ] 22 result JSONs + 44 metrics snapshots committed, each under its own
      workload folder
- [ ] Every run tagged: `ok` / `saturated` / `void(reason)`
- [ ] Every workload README regenerated and reflecting its final run set
- [ ] Variance-duplicate pair present for both engines
- [ ] If P3 exits by 5:30 → [P5](P5-stretch.md); else straight to
      [P4](P4-writeup.md)
