# Four defects found auditing the completed matrix, and the pre-registered re-run

After P3 and the text-only control were done, a defect audit of the finished
data (not the code) turned up four issues. Two are cosmetic-to-the-headline
(the affected cells were already excluded); two bear directly on whether the
clean comparison is admissible. All four are corrected by one pre-registered
re-run, staged in `scripts/{repilot,rerun_clean,graph_ab}.sh` and registered in
[SPEC §6](../../spec/SPEC.md) *before* the re-run fires. Verified facts first,
in the order they were checked.

## D3 — no restart between rate cells, and Cold wasn't cold above `0.5κ`

**Verified.** Prometheus counters are continuous across a workload's rate cells:
`vllm:prefix_cache_queries_total` reads `476720` at the end of Full-reuse
`0.5κ` and `476720` at the start of `0.8κ`; `1277650` → `1277650` at the next
boundary. SGLang's `sglang:prompt_tokens_total` shows the same continuity. So
the server was **not** restarted within a workload block — every rate cell
inherited the previous cell's warm caches.

The consequence is not hypothetical. `--no-enable-prefix-caching` disables
vLLM's KV prefix cache but leaves its **multimodal processor cache** (default
4 GiB, keyed by image content hash) running. Cold replays the *same*
`requests.jsonl` at each rate, so image #i at `0.8κ` is the same bytes the
processor cache already saw at `0.5κ`. Measured mm-cache hit rate on Cold/vLLM,
diffed pre→post per cell:

| Rate | mm queries | mm hits | hit rate |
|---|---|---|---|
| `0.5κ` | 250 | 0 | 0.0 % |
| `0.8κ` | 250 | 250 | **100.0 %** |
| `1.2κ` | 350 | 250 | **71.4 %** |

Cold was cold only at the lowest rate; above it, the vision encoder was being
skipped for most requests. Both `0.8κ` and `1.2κ` Cold cells were already
excluded as saturated, so **no reported number moves** — but the flag silently
did not mean what its name implies, and any future design leaning on Cold at a
higher rate would inherit the error.

Fix, two layers: Cold now also passes `--mm-processor-cache-gb 0` (vLLM) and
`SGLANG_VLM_CACHE_SIZE_MB=0` (SGLang) — the SGLang analogue is a 100 MB
hash-keyed embedding cache with the same exposure — **and** the re-run restarts
the server before every cell, which clears all cache state regardless of any
one flag's reach. The restart is the guarantee; the flags make "cold" true even
within a running server.

## D4 — knee miscalibrated, and n=1 at the only clean point

**Verified, with one correction to the premise.** The first pilot did *not* use
completion ratio *alone* — `scripts/pilot.py` also had an 8× TTFT-blowup
backstop. But an 8× blowup on a coarse geometric sweep is far laxer than the
`ttft_growth_ratio > 2.0` criterion the final data demanded, so the outcome is
exactly as described: `κ_sglang` was overestimated, `0.8κ` landed past-knee for
SGLang on Full/Partial reuse and for both engines on Cold, and only `0.5κ`
survived as clean — at **n=1**. The variance block (n=3) was spent entirely at
`0.8κ`/`1.2κ`, both saturated: it measured variance where the number is
meaningless and not where the verdicts are drawn. And measured completions at
the clean points are 150–372 (`completed − warmup`), so **p99 is unreportable
at every clean point** and reportable only at saturated ones — backwards.

Fix: `pilot.py` now uses `ttft_growth_ratio > 2.0` as the **primary** knee
signal (blowup kept only as a backstop for a point already uniformly slow on
arrival), so `κ` is calibrated by the same rule the matrix cells are judged by.
The re-run (`scripts/rerun_clean.sh`) samples two rates both below
`min(κ_vllm, κ_sglang)` per workload, n=3 each, with run length lifted so Full
and Partial reuse clear ≥1000 completions (a powered p99). Cold is wall-clock
capped and reports **p95, pre-registered** — ≥1000 Cold completions cost
~30 min/cell, and p99 for Cold is accepted as underpowered rather than chased.

## D2 — SGLang ran without prefill CUDA graphs; vLLM ran with them

**Verified against our own captures.** `gates/sglang/cuda_graph_info.txt`
contains the literal line *"Breakable CUDA graph is incompatible with
multimodal model; disabling prefill CUDA graph"* and then a decode-only capture;
`gates/vllm/cuda_graph_info.txt` shows 51 piecewise + 35 full shapes covering
prefill. So the matrix conflated engine architecture with a configuration
default. The source audit
([fairness-audit.md](fairness-audit.md)) confirmed the auto-disable is skipped
when the prefill backend is set explicitly, so the confound is directly
testable rather than merely inferable.

Fix: `scripts/graph_ab.sh` runs SGLang stock vs
`--cuda-graph-backend-prefill tc_piecewise` on Full/Partial reuse, n=3 at the
clean rate, restart between arms, startup logs preserved. Two admissible
outcomes: the forced arm runs and its delta *measures* the graph contribution
(a second, independent cross-check on the text-only split verdict), or it faults
on capture/replay — the documented reason the auto-disable ships for this
architecture — in which case the fault *is* the result and the asymmetry
reclassifies from "confound the study failed to control" to "engine property a
user of this model cannot avoid," which a defaults comparison is allowed to
absorb.

**Knee extension.** The stock re-pilot found SGLang's saturation knee is only
~45–55 % of vLLM's on every workload (full-reuse 1.88 vs 4.12 req/s; cold 0.72
vs 1.30) — all measured with prefill graphs *off* (the default). So a fixed-rate
TTFT delta is not the whole D2 question: does forcing prefill graphs on move the
*knee*? `scripts/repilot_sglang_graphs.sh` re-pilots SGLang with
`--cuda-graph-backend-prefill tc_piecewise` on full/partial reuse and compares
the knee against `pilot_sglang.json`. Knee jumps toward vLLM's → the graph
disable is the dominant cause of SGLang's saturation disadvantage; knee barely
moves → it is scheduling/KV, not graphs; capture faults → structural and
unavoidable for this model. It guards against a silent re-disable (server up but
log still says "disabling prefill CUDA graph" ⇒ the flag didn't take, knee is
effectively stock).

## D1 — one-sided JIT contamination (fixed at the tagging layer, found mid-re-run)

All four vLLM clean-rate cells carried `jit_contaminated`; all four SGLang
counterparts were clean — an exclusion perfectly correlated with engine
identity, the one confound an engine comparison cannot absorb. Direction of
every affected verdict was already safe (the contamination biases *against*
vLLM, the winner; the text-only n=3 blocks showed rep1 contaminated,
rep2/rep3 not, p50 differing ~1–2 %), but "safe direction" is not "admissible."

The re-run's first cell exposed the real problem, which longer warmup alone does
**not** fix: `check_jit_contamination.py` scanned the whole log slice
(warmup + measured) and flagged a JIT event wherever it fired. For these
fixed-shape workloads the `kernel_unified_attention` compile fires **once**, on
the first request of a shape — i.e. in warmup — and never again, so the measured
window is clean (rep1: p99 316 ms, `ttft_growth_ratio` 0.991, no spike) yet the
cell was still tagged `jit_contaminated`. Longer warmup (50) reliably pushes the
compile into warmup, but the check couldn't tell warmup JIT from measured JIT,
so it kept mis-tagging clean vLLM cells — D1, reborn.

Real fix, two parts: `client.py` now records `measured_start_epoch` (wall-clock
when the first non-warmup request is dispatched), and
`check_jit_contamination.py` parses each JIT log line's `MM-DD HH:MM:SS`
timestamp (UTC on these pods) and tags **only** events at/after that epoch (−2 s
guard). A warmup-absorbed compile is now correctly ignored; one that actually
lands in the measured window is still caught. Because the restart-per-cell
overwrites the engine log, this has to be right at run time — it cannot be a
post-hoc retag like the `ttft_growth` one — so it was fixed before restarting
the clean re-run (cost: one redone cell).

## Why one re-run, pre-registered

The four defects share a fix surface: a correctly-calibrated knee (D4) gives
clean rates; running n=3 at those rates with a restart before each cell (D3)
and a longer warmup (D1) and a graph A/B alongside (D2) closes all four in a
single ~6–7 h pod session. Registering the corrections in
[SPEC §6](../../spec/SPEC.md) before the session fires keeps the re-run
pre-registered rather than fitted to the numbers it will produce — the same
frozen-spec discipline every prior amendment followed.
