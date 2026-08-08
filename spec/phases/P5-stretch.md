# P5 — Stretch (conditional)

| | |
|---|---|
| **Objective** | Extra attribution, only from surplus |
| **Entry** | P3 exited by 5:30. Otherwise this phase does not exist today. |
| **Exit** | Any started item finished; nothing half-run |
| **Hard rule** | An incomplete core matrix with three half-run stretch workloads is worse than a clean core matrix. Do not start an item you cannot finish. |

## Priority order

Text only is first because it buys **attribution**: if engines diverge on
multimodal traffic but not on matched text-only traffic, the finding is about
the multimodal cache path specifically — which reframes the headline.

| # | Workload | Construction | Probes | Cost |
|---|----------|-------------|--------|------|
| 1 | Text only | Full reuse + Partial reuse replicated text-only, matched token counts, single mid rate | Is the divergence multimodal-specific? | 4 runs, ~20 min |
| 2 | Cache thrash | Working set of distinct prefixes ≈2× the engine's **reported** KV-token capacity (sized per engine — equal memory fraction ≠ equal capacity), Zipf-sampled | Eviction under thrash — where caches actually fail | 6 runs |
| 3 | Branching reuse | 8 images × 12 questions, interleaved | Tree-shaped reuse | 6 runs |
| 4 | Multi-turn chat | 4-turn conversations, full-history resend | Growing self-reuse; needs conversation state the harness may not support — highest schedule risk | 6 runs |

**Cache thrash is dropped if the Mooncake replay is committed to day 2**
([P6](P6-last-experiment.md)): an hour of real traffic against finite KV *is*
the eviction experiment, under a real working set instead of a Zipf synthetic.

Each stretch workload gets its own folder and generated README on the same
terms as the core four ([SPEC §8](../SPEC.md)), and every stretch run follows
the [P3](P3-core-matrix.md) per-run procedure — scrapes, validity checks, tags,
re-render — without exception. Stretch data that skipped the validity gate is
not data.

## Exit checklist

- [ ] Started items completed and tagged, or the phase was skipped cleanly
- [ ] Any stretch findings quarantined in a separate README subsection from the
      pre-registered H1–H7 results
