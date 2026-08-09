# P4 — Writeup (6:00–8:00)

| | |
|---|---|
| **Objective** | Shippable repo: two figures, filled results, H7 predictions committed |
| **Entry** | P3 exit (P5 optional in between) |
| **Exit** | Pushed. H7 prediction commit exists *before* any P6 run |
| **Fallback** | Cut gap.png before cutting the Results text; never cut the H7 prediction commit |

> **Amended after the real P3 matrix ran.** `0.8κ` did not survive as a clean
> comparison point for Full reuse, Partial reuse, or Cold — see
> [SPEC §3](../SPEC.md)'s amendment note and
> [`../../learnings/measurement/ttft-growth-signal.md`](../../learnings/measurement/ttft-growth-signal.md).
> This phase's figures, mechanism bullet, and results template are all
> written against the corrected picture below, not the originally intended
> one — read this whole page before drafting, not just the figures section.

## Figures — exactly two, both need to say more than they used to

1. **`figures/ttft.png`** — p50 TTFT vs offered rate, one line per engine,
   three panels (Full reuse / Cold / Partial reuse). Saturated points hollow.
   **The TTFT axis must be log-scaled, and the axis must say so.** The real
   range now spans ~227ms (Full reuse, vLLM, `0.5κ`) to ~180,000ms (Cold,
   SGLang, `1.2κ`) — over three orders of magnitude. On a linear axis the
   entire healthy region, where the actual comparison lives, is a flat line
   on the floor and the plot visually screams "overload" instead of showing
   the sub-second differences that are the real result. Hollow-for-saturated
   does not fix a linear axis at this range; the log scale does.
2. **`figures/gap.png`** — engine TTFT delta (%) vs measured cached-token
   fraction, **faceted by rate, sub-knee runs only**. The facet condition is
   now **per-engine-per-workload, not per-rate** — "sub-knee" is engine-
   dependent (SGLang is past-knee at `0.8κ` on Full reuse and Partial reuse
   where vLLM is not; both engines are past-knee at `0.8κ` on Cold). A cell
   only qualifies as a clean point if *both* engines being compared were
   sub-knee at that rate. With `0.8κ` disqualified everywhere it doesn't
   survive, this leaves very few clean points — likely one rate's worth
   (`0.5κ`) per workload. **Decide explicitly which of these before drawing
   it:**
   - Plot the sparse points as-is with an explicit "n points, not a fitted
     trend" label on the figure itself, or
   - Demote the cache-effect claim to a table with a stated caveat that
     saturation prevented a clean multi-rate sweep.

   Do not let the figure imply a resolved cache-vs-rate curve it no longer has
   the points to support — this directly affects whether H7's prediction
   (below) is reading off a real fitted relationship or off two points
   connected by a line that looks like one.

Single stream, VisionArena replay, and the variance cell report as tables, not
plots.

Figures stay top-level: each spans several workloads, so it belongs to none of
them. Per-workload results already live in each workload's generated README
([SPEC §8](../SPEC.md)) — the top-level README does not repeat those tables, it
links to them and states the verdicts.

## Results template (fills the top-level README.md)

For each of H1–H7: **confirmed / falsified / inconclusive**, the measured
delta, and the variance-cell context next to it. Include nulls — and use the
**two distinct null categories** from [SPEC §3](../SPEC.md#3-hypotheses-frozen-at-the-p2-commit):
"measured at a clean point, no effect" is a different, weaker claim than "no
clean comparison point survived saturation," and the template must not let
the second get laundered into the first. Several of H1/H4a/H4b will land in
the second category now that `0.8κ` is disqualified for most workloads.

Then, exactly four bullets:

- Gate B ratios per engine, and what they say the multimodal path caches
  (KV vs encoder — this may itself be the headline)
- **Mechanism for the observed Partial-reuse (and Full-reuse) delta, argued
  as a fork between two competing primary hypotheses, not cache-effect
  checked first with graphs as a footnote.** Given the real data — SGLang
  collapsing at `0.8κ` on Full reuse while vLLM stays clean — the two
  candidate explanations are (a) a genuine cache/scheduling difference and
  (b) SGLang paying un-graphed prefill launch overhead vLLM doesn't, since
  SGLang auto-disables CUDA-graph capture for prefill on this multimodal
  model while vLLM graphs prefill-covering batches
  ([`../../learnings/infrastructure/cuda-graphs.md`](../../learnings/infrastructure/cuda-graphs.md)).
  Adjudicate with the counters already collected — cached-token fraction,
  preemption counts, `gates/<engine>/cuda_graph_info.txt` — not vibes, and not
  a default assumption that caching explains the gap unless graphs are ruled
  out. (b) is independently interesting even if it turns out to dominate: a
  real, publishable difference in multimodal prefill handling that has
  nothing to do with prefix reuse.
- **What the saturated runs showed, interpreted as queueing behavior — and
  use the one real replicate set as evidence, not just an annotation.** Full
  reuse `0.8κ` has n=3 on both engines (the late-pass variance block): SGLang's
  three reps read 10.4s / 12.5s / 11.9s p50 — roughly 20% spread at what the
  design intended as a "healthy" rate. That spread is itself the argument for
  why `0.8κ` cannot be the comparison point: near-knee measurement is this
  unstable, not just slower.
- The H7 predictions read off gap.png — committed before any Mooncake-replay
  run — and, once day 2 lands, predicted vs observed with an honest miss
  analysis. Per [P6](P6-last-experiment.md)'s saturation gate: a miss must be
  attributed to either a cache-model failure or the replay running past the
  knee, not left ambiguous.

## The H7 prediction commit

Non-negotiable ordering, the whole value of [P6](P6-last-experiment.md):

1. Read the predicted engine gap off gap.png at 40% and 59% cached fraction
   (Mooncake Conversation and Tool&Agent reuse ratios) — **and state the load
   regime (rate relative to κ) those points were read from**, since the
   prediction is only valid if the replay's own offered load lands in a
   comparable regime ([P6](P6-last-experiment.md))
2. Write both numbers into README.md
3. Commit
4. Only then may any Mooncake-replay run start — prediction and observation
   live in separate commits

## Known limitations — state verbatim in the README

N=1 hardware, one model, one day; synthetic noise images (encoder cost is
content-independent, but real traffic has correlated prefix structure this
workload only approximates); open-loop Poisson arrivals (addressed on day 2 by
the Mooncake replay's preserved trace timestamps); p99 underpowered at n=200;
engine scheduler defaults recorded, not swept; **the shared rate grid's `0.8κ`
point, calibrated from a coarse pilot sweep, did not survive as a clean
sub-saturation comparison for most workloads — the real comparison is
narrower than pre-registered** ([SPEC §3](../SPEC.md)). This list is the
difference between a benchmark and a blog post.

## Exit checklist

- [ ] Both figures render and are committed; `ttft.png`'s axis is log-scaled
      and labelled as such
- [ ] `gap.png`'s facet logic is per-engine-per-workload, and its sparse-data
      caveat (or table demotion) is stated on the figure/in the README, not
      implied
- [ ] Mechanism bullet treats cache-effect and graph-effect as adjudicated,
      not assumed
- [ ] Every workload README regenerated after the final run
- [ ] Top-level README results section complete, both null categories used
      where they apply
- [ ] H7 prediction commit exists, its hash is noted, and its stated load
      regime is recorded
- [ ] Repo pushed
