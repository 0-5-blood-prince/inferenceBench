# inferenceBench — writeup briefing

Paste this whole file as the opening prompt in a new thread to work on the
P4 writeup. It is a complete status dump: what the study is, what ran, what
the real numbers say, and what's still open. Repo: `/Users/mabdul/inferenceBench`,
branch `main`, HEAD `f7ec7ed` as of this dump.

## The question

Does prefix reuse change which serving engine wins — vLLM or SGLang — under
multimodal prefix-caching? One A100 80GB, one model
(`google/gemma-4-31B-it`), pinned versions `vllm==0.26.0` / `sglang==0.5.16`.
Full pre-registration is `spec/SPEC.md`; the phase runbook is
`spec/phases/P0..P6-*.md`.

## Where things stand, phase by phase

- **P0 (feasibility)** — done. Locked model, probed image tokens (Gemma 4:
  fixed 258 tokens at every resolution — not a dynamic-resolution ViT), built
  the four core workloads.
- **P1 (gates)** — done. Gate A (strict output-equivalence) was tried and
  **dropped as a blocker** — it tests a property no latency comparison needs
  and manufactured a false engine asymmetry from bf16 reduction-order noise
  (`learnings/gate-a/`). Replaced by Gate R (reuse-by-counters + budget
  pinning + shape matching). Gate B (below) is the surviving cache-layer
  probe.
- **P2 (freeze)** — done. Piloted **both** engines (not just one) to find the
  saturation knee κ per workload, took `κ = min(κ_vllm, κ_sglang)`. Rate grid
  frozen in `run.sh:rate_for()`.
- **P3 (core matrix)** — done, 28/28 runs. See results below. Took ~17.8h
  wall-clock (spec's "exited by 5:30" entry condition for P5 was blown by a
  wide margin — see P5 disposition below).
- **P4 (writeup)** — **in progress, this is the phase to work in.**
  `spec/phases/P4-writeup.md` has the up-to-date figure spec, mechanism
  bullet (now filled with real text-only-control numbers), results template,
  and exit checklist. Nothing in the top-level `README.md` has been written
  yet — that's the actual writeup task.
- **P5 (stretch)** — **skipped as written**, on its own entry condition
  (`spec/phases/P5-stretch.md` has the full reasoning + each item's
  redirected fate). Text-only was promoted into P4 and has since been run
  for real (see below). Cache thrash dropped (subsumed by Mooncake replay).
  Branching reuse / multi-turn chat pushed to a "future work" section in P4.
- **P6 (last experiment)** — not started. VisionArena replay currently
  blocked on RunPod capacity (pod stop/resume cycle wiped the container disk;
  waiting for a free A100 to resume pod `q056e1bfufki5u`). Mooncake replay
  needs the H7 prediction commit from P4 first — cannot start until P4's
  gap-vs-cached-fraction figure exists.

## The real P3 core-matrix numbers (28 runs, all retagged)

Four workloads × up to 3 rates (`0.5κ`, `0.8κ`, `1.2κ`) × 2 engines, plus a
late-pass variance block (Full reuse @ `0.8κ`/`1.2κ`, both engines, 2 extra
reps → n=3) and Single stream (concurrency=1, one rep/engine).

**p50 TTFT (seconds), tag in brackets:**

| Workload | Rate | vLLM | SGLang |
|---|---|---|---|
| Full reuse | 0.5κ | 0.227 [ok, jit] | 0.341 [ok] |
| Full reuse | 0.8κ (n=3) | 0.285–0.289 [ok] | 10.4–12.5 [**saturated**] |
| Full reuse | 1.2κ (n=3) | 30.9–32.4 [saturated] | 74.0–76.5 [saturated] |
| Cold | 0.5κ | 0.573 [ok, jit] | 1.000 [ok] |
| Cold | 0.8κ | 1.058 [**saturated**] | 20.6 [saturated] |
| Cold | 1.2κ | 44.0 [saturated] | 96.2 [saturated] |
| Partial reuse | 0.5κ | 0.329 [ok, jit] | 0.561 [ok] |
| Partial reuse | 0.8κ | 0.426 [ok] | 22.4 [**saturated**] |
| Partial reuse | 1.2κ | 48.9 [saturated] | 90.9 [saturated] |
| Single stream | n/a, conc=1 | 0.138 [ok, jit] | 0.251 [ok] |

**Only `0.5κ` survives as a clean (both-engine-healthy) comparison point for
every workload.** `0.8κ` looked clean on completion-ratio alone but SGLang
had already collapsed under queueing on Full reuse and Partial reuse — this
was caught by adding a second saturation signal (`ttft_growth_ratio`, median
TTFT of a run's second half vs first half; threshold >2.0) and retagging all
28 results in place, no re-run needed. 16 of 28 cells flipped `ok`→
`saturated` this way. Full detail:
`learnings/measurement/ttft-growth-signal.md`.

**Gate B** (does a second identical request reuse KV faster? — ratio =
ttft2/ttft1): vLLM 0.661, SGLang 0.765. Both partial, neither near 0 or 1 —
reading is "one cache layer only (KV without vision tower, or reverse)," not
a failure. Raw: `gates/{vllm,sglang}/gate_b_result.json`.

**CUDA graph capture asymmetry** (`learnings/infrastructure/cuda-graphs.md`):
SGLang auto-disables prefill graph capture for this multimodal model; vLLM
graphs prefill-covering batches. Both engines independently chose Triton for
the attention kernel (ruled out as an explanation). This was the leading
candidate mechanism for the SGLang TTFT disadvantage — until the text-only
control below partially adjudicated it.

## Text-only control — run for real, result is a split verdict

Promoted out of P5 into P4 as a first-class control (not squeezed into
surplus time): Full reuse and Partial reuse replicated with the image
stripped and replaced 1:1 by an equal-token-count text pad, at each
workload's own `0.5κ` rate, n=3/engine, both engines clean
(`completion_ratio: 1.0` on all 12 cells).

| Workload | vLLM p50 (ms) | SGLang p50 (ms) | Gap, multimodal | Gap, text-only |
|---|---|---|---|---|
| Full reuse | 218.8 | 290.7 | 50.5% | **32.9%** |
| Partial reuse | 316.6 | 507.3 | 70.7% | **60.2%** |

The gap shrinks by roughly a third on both workloads but does **not**
collapse to noise on either. That rules out "the gap is purely a
multimodal-prefill/CUDA-graph artifact" (would've gone to ~0) and rules out
"caching explains all of it" (would've stayed flat) — **it's a split: the
CUDA-graph-capture asymmetry accounts for something like a third of the gap,
a general scheduling difference accounts for the rest.** Full detail and raw
numbers: `workloads/full-reuse-text/README.md`,
`workloads/partial-reuse-text/README.md`. This is now written into
`P4-writeup.md`'s mechanism bullet.

## What P4 (the writeup) still needs — this is the actual task

Read `spec/phases/P4-writeup.md` in full before starting; it has been amended
several times and the current version reflects everything above. In order:

1. **`figures/ttft.png`** — p50 TTFT vs offered rate, one line/engine, three
   panels (Full reuse/Cold/Partial reuse), saturated points hollow,
   **log-scaled TTFT axis, labelled as such** (real range ~227ms–~180,000ms).
2. **`figures/gap.png`** — TTFT delta vs cached-token fraction, faceted
   **per-engine-per-workload** (not per-rate — sub-knee is engine-dependent).
   With `0.8κ` disqualified almost everywhere, this is sparse (~1 rate's
   worth per workload) — the spec requires either an explicit "n points, not
   a fitted trend" label on the figure, or demoting the claim to a table.
   Decide which before drawing it.
3. **Results template** in the top-level `README.md`: H1–H7 each rated
   confirmed/falsified/inconclusive, using the **two distinct null
   categories** (SPEC §3) — "measured clean, no effect" vs "no clean point
   survived saturation" are not interchangeable. Several of H1/H4a/H4b will
   land in the second category.
4. **Four required bullets**: Gate B reading; the mechanism fork
   (cache-effect vs graph-effect, now with the text-only split-verdict result
   filled in — see above); the saturated-run/queueing interpretation (use
   the n=3 Full-reuse `0.8κ` spread — SGLang 10.4/12.5/11.9s, ~20% — as
   direct evidence for why `0.8κ` isn't a usable comparison point, not just
   an annotation); the H7 prediction-commit bullet (blocked on P6 Mooncake
   replay planning, but the *predictions* need to be read off `gap.png` and
   committed before any Mooncake run starts — non-negotiable ordering, see
   P4's "H7 prediction commit" section).
5. **Future work section** (already scaffolded in P4-writeup.md): branching
   reuse and multi-turn chat, dated, multi-turn's harness prerequisite
   (no conversation-state support yet) stated explicitly.
6. **Known limitations** section — verbatim text is already drafted in
   `P4-writeup.md`, just needs to land in the real README.

## Known open items / risks for whoever picks this up

- **P6 VisionArena/Mooncake replay has not run yet.** Pod capacity has been
  intermittent (RunPod host `j9i7o9xcsz1o` repeatedly returns "not enough
  free GPUs" on resume attempts) — if writing the H7 prediction section,
  remember the ordering rule: predictions committed *before* any Mooncake run.
- **RunPod stop/resume wipes the container disk** (venvs, apt packages;
  network volume persists). Any future pod resume needs `./run.sh bootstrap`
  rerun and `ninja-build` reinstalled via apt (undeclared dependency in both
  engines' pip metadata — same class of gap as the original P0 finding).
- GPU clocks cannot be locked from inside the container (no permission) —
  thermal drift is uncontrolled, mitigated only by engine-order alternation
  and the late-pass variance block.
- All spec amendments after the P2 freeze commit are logged as stated-reason
  blockquotes in the relevant `spec/phases/*.md` file, never silent edits —
  worth skimming those blockquotes for anything that changes how a figure or
  claim should be read.

## File map

- `spec/SPEC.md` — frozen pre-registration, hypotheses H1–H7, all amendments
- `spec/phases/P0..P6-*.md` — the runbook, current status per phase above
- `workloads/{full-reuse,cold,partial-reuse,single-stream}/` — core matrix,
  each with `results/`, `metrics/`, generated `README.md`
- `workloads/{full-reuse,partial-reuse}-text/` — text-only control, `README.md`
  has the numbers table
- `gates/{vllm,sglang}/` — P1 gate artifacts (Gate B, Gate C, config dumps,
  CUDA graph capture log greps)
- `learnings/` — 13 topical modules on everything that went wrong or was
  learned operationally; `LEARNINGS.md` is the index
- `run.sh` — orchestration (bootstrap/up/down/cell/p3); `scripts/client.py` —
  load generator; `scripts/pilot.py` — κ finder; `scripts/render_readme.py` —
  per-workload README generator
