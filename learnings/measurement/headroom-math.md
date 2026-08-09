# Row headroom: why the client can run dry, and how it's asserted rather than trusted

`client.py`'s loop can never run longer than the number of request rows it
loaded — once it exhausts them, it stops, regardless of what the intended stop
condition (`_done()`) says. Rows are a hard ceiling the time floor cannot
override. Get the row count wrong and the loop silently reports a shorter run
than intended, with no error, because "ran out of rows" isn't a crash — it's
just the end of a list.

This is exactly what happened in the bug documented in
[p3-statistical-review.md](p3-statistical-review.md)'s harness follow-up: `run.sh
cell()` never scaled `--num-requests` to the cell's actual rate, so at the P2
grid's higher rates (e.g. 4.899 req/s) the default row count arrived in ~51s
against an intended 240s floor, and every affected cell would have reported a
clean `ok` result that was actually a third of its intended length.

## The fix: size the row pile with headroom, not exactly to the target

```python
min_needed = warmup + int(rate * min_seconds * 1.15)
```

- **`rate × min_seconds`** is the *expected* number of arrivals in the target
  window — at 4.899 req/s over 240s, ~1176.
- **`× 1.15`** is margin, not precision. Arrivals are Poisson, not clockwork:
  the actual count in any run wobbles around that expected value with std dev
  ≈ √1176 ≈ 34. An unlucky run could land arrivals meaningfully short of the
  mean; 15% headroom (~176 extra rows here) sits well outside that wobble, so
  ordinary variance can't starve the loop right at the finish line.
- **`+ warmup`** because discarded warmup requests consume rows too, before
  the clock even starts.

Verified live on this pod at the exact rate that broke: `50 + int(4.899 × 240
× 1.15) = 1402` rows requested; the real run consumed 1131 completions over
331.9s, comfortably inside budget without over-building.

**Why 1.15 in `client.py` but 1.1 in `pilot.py`:** different jobs, different
cost of overshoot. Overshoot only costs time in one direction — demanding
*more* rows than needed makes the loop wait longer for that count to arrive,
never shorter (the inverse mistake, once made in `pilot.py` at 2.5x, described
in [pilot-methodology.md](pilot-methodology.md)). `pilot.py` runs 5-8 short
120s points back to back per sweep, so a 10% overshoot per point is kept tight
because it accumulates in wall-clock cost across the sweep. `client.py`
protects one long 240s+ measurement per matrix cell — no stack of points to
accumulate overshoot across, so a slightly fatter 15% cushion is cheap
insurance against the Poisson tail. Neither number is derived from anything
more precise than "comfortably above the mean, not wasteful" — and that's
legitimate specifically *because* row count feeds no metric, threshold, or
hypothesis; it only decides how much ammunition is on hand.

## The margin is not a guarantee, so it's asserted, not trusted

A margin can still be insufficient — an unusually bursty arrival pattern, or a
workload rebuilt with a lower `--max-rate` than the cell actually needs. Silent
truncation must not be possible to miss after the fact, so the loop now
detects its own termination cause directly: `for/else` on the request loop
means the `else` clause runs only when the loop completes **without** hitting
`break` — precisely "ran out of rows before the intended stop condition ever
fired." When that happens, the run is tagged `rows_exhausted` and excluded from
pooled analysis, the same way `saturated` and `jit_contaminated` are
([P3-core-matrix.md](../../spec/phases/P3-core-matrix.md)).

Verified locally (no GPU needed, pure control-flow logic) against both cases
before shipping: a too-small row pile correctly flags `rows_exhausted=True`;
a normal run that stops via `_done()` correctly reports `False`.

This is the same lesson as the JIT/warmup assertion from the P3 statistical
review: the bug is never really "the margin was wrong" — margins are always
approximate. The bug is a short run being indistinguishable from a good one
after the fact. The fix in both cases is the same shape: detect the failure
mode directly and tag it, rather than trust that the mitigation upstream of it
always works.

## Decided: `jit_contaminated`'s scope stays conservative, not tightened

`check_jit_contamination.py`'s scan window starts before `client.py` even
begins, so it covers the discarded warmup phase as well as measurement — it
cannot currently distinguish "JIT fired during warmup, harmless, exactly what
warmup is for" from "JIT fired during measurement, corrupts the numbers."
Three live tests all showed the event firing within ~2 seconds of server
startup and never recurring within the same session, strongly suggesting
these are warmup-phase events being (correctly) over-flagged, not proof of
corrupted data.

The quantified cost of leaving this as-is: the event is one-per-fresh-server-
start, and P3 has ~10 fresh starts (2 each for Full reuse, Cold, Partial
reuse, Single stream, plus the late-pass block). If it fires on the first
cell of most blocks — plausible given 3-for-3 so far — up to ~10 of 30 cells
could be excluded. Worst case is Single stream specifically: one run per
engine, no rate sweep, so a hit there is a total loss for that block, not a
missing first point.

Decided to accept this rather than build a tighter post-warmup-only window
(which would need `client.py` to mark when warmup ends and correlate that
wall-clock timestamp against the engine log's own timestamps — a real fix,
but one requiring its own live verification before trusting it). Excluded
cells are cheap to re-run: they are always the *first* cell of a block that
just started, so no extra engine restart is needed, and the cost sits inside
the OOM-slack budget already reserved in [P3-core-matrix.md](../../spec/phases/P3-core-matrix.md).
