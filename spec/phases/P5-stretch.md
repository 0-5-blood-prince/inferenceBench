# P5 — Stretch (conditional)

| | |
|---|---|
| **Objective** | Extra attribution, only from surplus |
| **Entry** | P3 exited by 5:30. Otherwise this phase does not exist today. |
| **Exit** | Any started item finished; nothing half-run |
| **Hard rule** | An incomplete core matrix with three half-run stretch workloads is worse than a clean core matrix. Do not start an item you cannot finish. |

> **This phase was skipped, not run, and the entry condition is exactly why.**
> P3 took ~17.8 hours of wall-clock, nowhere near "exited by 5:30" — by this
> phase's own hard rule, it does not exist for this run. Respecting a gate you
> wrote yourself, rather than reopening it because there's appetite for more
> workloads, is the same discipline that has governed every other amendment
> in this project.
>
> There is a stronger reason underneath the mechanical one: **P3's own data
> made this phase's premise obsolete before the clock even mattered.** P5 was
> designed to add attribution on top of a clean engine comparison. P3 didn't
> deliver one — `0.8κ` collapsed asymmetrically
> ([SPEC §3](../SPEC.md)'s amendment,
> [`../../learnings/measurement/ttft-growth-signal.md`](../../learnings/measurement/ttft-growth-signal.md)),
> Full reuse and Partial reuse survive as clean comparisons only at `0.5κ`, and
> a CUDA-graph confound is competing with the cache story for the mechanism.
> Running more workloads on top of a core that isn't yet fully interpreted is
> the same failure mode this phase's own hard rule names, one level up: an
> un-interpreted core with new stretch data on top of it is worse than a
> clean, fully-interpreted core. Finish the story underneath before adding
> surface area on top of it.
>
> The four items below are not abandoned — each has a stated fate:
>
> - **Text only (#1) was promoted, not skipped.** P3 handed this item a job it
>   didn't have when the priority order below was written: a live candidate
>   mechanism (SGLang disabling prefill CUDA-graph capture for this multimodal
>   model) that text-only is the exact discriminator for — matched token
>   counts, no image tokens, no encoder, no multimodal prefill path at all. If
>   the gap vanishes on text-only, the mechanism is multimodal-specific; if it
>   persists, it's general scheduling. That is no longer stretch attribution,
>   it is the control that decides a possibly-headline finding. Moved into
>   [P4](P4-writeup.md) as a first-class, properly designed experiment (0.5κ,
>   n≥3, not squeezed into surplus P3 minutes) — see P4's "Text-only control"
>   section.
> - **Cache thrash (#2) is dropped, per this file's own already-stated
>   condition** — the Mooncake replay is committed to day 2
>   ([P6](P6-last-experiment.md)), and an hour of real traffic against finite
>   KV *is* the eviction experiment, on a real working set instead of a Zipf
>   synthetic. The real-trace version dominates; state that in the writeup
>   rather than running the weaker synthetic version anyway.
> - **Branching reuse (#3) and multi-turn chat (#4) are real, distinct cache
>   patterns worth studying — just not today.** Multi-turn specifically needs
>   conversation-state harness support this repo doesn't have yet (its own
>   entry in the table below already named this the highest schedule risk).
>   Building unvalidated harness features on top of an uninterpreted core is
>   how afternoons die. Captured as dated future work in the top-level
>   README's template ([P4](P4-writeup.md)) instead of half-run today.

## Priority order (as originally designed — kept for the record, not executed)

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

- [x] Phase skipped cleanly — entry condition false, reason stated above, no
      item started
- [x] Text only relocated to [P4](P4-writeup.md) as a designed control, not a
      stretch item
- [x] Cache thrash confirmed absorbed by the Mooncake replay ([P6](P6-last-experiment.md))
- [x] Branching reuse and multi-turn chat recorded as dated future work in the
      top-level README template ([P4](P4-writeup.md))
