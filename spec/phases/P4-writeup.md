# P4 — Writeup (6:00–8:00)

| | |
|---|---|
| **Objective** | Shippable repo: two figures, filled results, H7 predictions committed |
| **Entry** | P3 exit (P5 optional in between) |
| **Exit** | Pushed. H7 prediction commit exists *before* any P6 run |
| **Fallback** | Cut gap.png before cutting the Results text; never cut the H7 prediction commit |

## Figures — exactly two

1. **`figures/ttft.png`** — p50 TTFT vs offered rate, one line per engine,
   three panels (Full reuse / Cold / Partial reuse). Saturated points hollow.
   The Partial-reuse panel is the headline.
2. **`figures/gap.png`** — engine TTFT delta (%) vs measured cached-token
   fraction, **faceted by rate, sub-knee runs only** — pooling across rates
   would let queueing effects masquerade as cache effects.

Single stream, VisionArena replay, and the variance cell report as tables, not
plots.

Figures stay top-level: each spans several workloads, so it belongs to none of
them. Per-workload results already live in each workload's generated README
([SPEC §8](../SPEC.md)) — the top-level README does not repeat those tables, it
links to them and states the verdicts.

## Results template (fills the top-level README.md)

For each of H1–H7: **confirmed / falsified / inconclusive**, the measured
delta, and the variance-cell context next to it. Include nulls.

Then, exactly four bullets:

- Gate B ratios per engine, and what they say the multimodal path caches
  (KV vs encoder — this may itself be the headline)
- Mechanism for the observed Partial-reuse delta or its absence, argued from
  cached-token counters, preemption counts, and queue/prefill decomposition —
  not vibes
- What the saturated runs showed, interpreted as queueing behavior, separately
- The H7 predictions read off gap.png — committed before any Mooncake-replay
  run — and, once day 2 lands, predicted vs observed with an honest miss
  analysis

## The H7 prediction commit

Non-negotiable ordering, the whole value of [P6](P6-last-experiment.md):

1. Read the predicted engine gap off gap.png at 40% and 59% cached fraction
   (Mooncake Conversation and Tool&Agent reuse ratios)
2. Write both numbers into README.md
3. Commit
4. Only then may any Mooncake-replay run start — prediction and observation
   live in separate commits

## Known limitations — state verbatim in the README

N=1 hardware, one model, one day; synthetic noise images (encoder cost is
content-independent, but real traffic has correlated prefix structure this
workload only approximates); open-loop Poisson arrivals (addressed on day 2 by
the Mooncake replay's preserved trace timestamps); p99 underpowered at n=200;
engine scheduler defaults recorded, not swept. This list is the difference
between a benchmark and a blog post.

## Exit checklist

- [ ] Both figures render and are committed
- [ ] Every workload README regenerated after the final run
- [ ] Top-level README results section complete, nulls included
- [ ] H7 prediction commit exists and its hash is noted
- [ ] Repo pushed
