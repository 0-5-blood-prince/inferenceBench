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

Single stream, VisionArena replay, the variance cell, and the text-only
control report as tables, not plots.

Figures stay top-level: each spans several workloads, so it belongs to none of
them. Per-workload results already live in each workload's generated README
([SPEC §8](../SPEC.md)) — the top-level README does not repeat those tables, it
links to them and states the verdicts.

## Text-only control experiment

> **Added after P5 was skipped as written** ([P5](P5-stretch.md)) — this item
> was originally P5 priority #1, run only from surplus time. P3's real
> results gave it a job P5 never had: it is now the direct discriminator
> between the two competing primary hypotheses in the mechanism bullet below
> (genuine cache/scheduling difference vs. SGLang's un-graphed multimodal
> prefill overhead). A discriminator for a possibly-headline finding deserves
> a properly sized, designed run, not whatever surplus minutes P3 left behind
> — so it moved here as first-class, not squeezed into P5's original 4-run/
> 20-min budget.

- **Construction:** Full reuse and Partial reuse workloads, replicated with
  the image stripped out and prompt text padded to match each workload's
  original total token count — same prefix-reuse structure, zero image
  tokens, no vision-encoder path.
- **Rate:** `0.5κ` only — the one rate that stayed clean for both engines on
  every core workload ([SPEC §3](../SPEC.md) amendment). This experiment is
  worthless as a discriminator if either engine is queueing.
- **Replicates:** n≥3 per engine per workload (matches the late-pass variance
  block's design, not the n=1 default), so a text-only gap or its absence is
  read against the same ~20%-spread near-knee instability already found in
  the real data, not mistaken for noise.
- **Reading it:** if the vLLM/SGLang TTFT gap seen on Full reuse and Partial
  reuse shrinks to noise on their text-only twins, the gap is multimodal-
  prefill-specific — evidence for the CUDA-graph-capture asymmetry
  ([`../../learnings/infrastructure/cuda-graphs.md`](../../learnings/infrastructure/cuda-graphs.md)).
  If the gap persists essentially unchanged, it's a general scheduling
  difference unrelated to images, and the cache-effect explanation regains
  ground. Either outcome is reported in the mechanism bullet below, not left
  as a separate orphaned finding.
- **Cost:** 2 workloads × 2 engines × n≥3 ≈ 12 runs, ~30–40 min at `0.5κ`
  generation lengths — larger than P5's original 4-run estimate because
  the replicate count now matches the variance design, not surplus-time
  economy.
- Gets its own folder and generated README on the same terms as the core
  four ([SPEC §8](../SPEC.md)); reports as a table here, not a figure — four
  cells (2 workloads × 2 engines) is too sparse to plot honestly.

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
  preemption counts, `gates/<engine>/cuda_graph_info.txt` — plus the
  text-only control above, whose whole purpose is to strip out (b) by
  construction. Not vibes, and not a default assumption that caching explains
  the gap unless graphs are ruled out. (b) is independently interesting even
  if it turns out to dominate: a real, publishable difference in multimodal
  prefill handling that has nothing to do with prefix reuse.
  **Result (text-only control, n=3/engine, ran after P5 was skipped):** the
  gap shrinks but does not vanish on either workload — Full reuse 50.5%→32.9%,
  Partial reuse 70.7%→60.2%
  ([`../../workloads/full-reuse-text/README.md`](../../workloads/full-reuse-text/README.md),
  [`../../workloads/partial-reuse-text/README.md`](../../workloads/partial-reuse-text/README.md)).
  Both workloads shrinking by roughly the same fraction, neither dropping to
  noise, is a split verdict, not a winner: (b) accounts for something like a
  third of the gap and (a) or another general-scheduling effect accounts for
  the rest. Write the mechanism bullet as a split finding, not an
  either/or — the text-only control is what makes the split visible instead
  of assumed.
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

## Future work — state verbatim in the README

Two workloads from P5's original priority order are not run this cycle and
are recorded here as dated future work, not silently dropped
([P5](P5-stretch.md)):

- **Branching reuse** (8 images × 12 questions, interleaved — tree-shaped
  reuse rather than the linear reuse the core four test) — not run as of
  this writeup. No harness blocker; deferred purely for schedule reasons
  once P5's entry condition failed.
- **Multi-turn chat** (4-turn conversations, full-history resend — growing
  self-reuse within a session) — not run as of this writeup, and unlike
  branching reuse this one has a real prerequisite: the day-1 harness has no
  conversation-state support, so this needs new, unvalidated harness work
  before it can run at all. Flagged in P5 as the highest schedule risk of the
  four original stretch items; that risk is exactly why it's deferred rather
  than rushed.

Cache thrash is not listed here because it isn't deferred — it's subsumed:
the Mooncake replay ([P6](P6-last-experiment.md)) is a strictly better
version of the same eviction question, run against a real working set
instead of a Zipf synthetic.

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
- [ ] Text-only control run: 2 workloads × 2 engines × n≥3 at `0.5κ`, its own
      folder README regenerated, result table in the top-level README
- [ ] Mechanism bullet treats cache-effect and graph-effect as adjudicated
      using the text-only control, not assumed
- [ ] Every workload README regenerated after the final run
- [ ] Top-level README results section complete, both null categories used
      where they apply
- [ ] Future work section lists branching reuse and multi-turn chat, dated,
      with multi-turn's harness prerequisite stated
- [ ] H7 prediction commit exists, its hash is noted, and its stated load
      regime is recorded
- [ ] Repo pushed
