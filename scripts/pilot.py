#!/usr/bin/env python3
"""P2 pilot: find one workload's saturation knee kappa on one engine.

    pilot.py --workload full-reuse --engine vllm --base-url http://localhost:8000 \
        --model google/gemma-4-31B-it

2-minute open-loop sweeps at geometrically increasing rates (SPEC section 2,
P2-freeze.md), stopping at the first point that looks saturated by ANY of three
signals:

  - ttft_growth_ratio within one point exceeds --ttft-growth (PRIMARY): the
    2nd-half/1st-half median-TTFT ratio, the same criterion the P3 retag used
    after completion-ratio alone missed 16 of 28 saturated cells
  - completed/offered drops below --threshold (SPEC's own stated definition)
  - p50 TTFT grows past --ttft-blowup times the first point's baseline TTFT
    (backstop for a point that is already uniformly slow on arrival)

The first calibration of kappa used completion-ratio + blowup only. That let
0.8k land past-knee for SGLang on Full reuse and Partial reuse, and for both
engines on Cold - the whole reason only 0.5k survived the matrix as clean. A
re-pilot with the growth signal as PRIMARY is what closes that gap; the knee it
finds is judged by the same rule the matrix cells are.
Measured on this pod: completion_ratio stayed at a perfect 1.000 while p50 TTFT
went from ~200-500ms to 19-31 SECONDS between one rate step and the next, and
the run after that blew the subprocess timeout entirely, taking down the whole
sweep with an unhandled exception - losing every point already collected, since
results were only written at the very end. That is not a spec deviation: SPEC
section 6 already says "above the saturation knee, open-loop TTFT is a function
of run length (the queue grows without bound)" - unbounded TTFT growth under
open-loop IS the definition of past-knee, and catching it before the queue
depth becomes unrecoverable is a more faithful reading of that sentence than
waiting for completion_ratio to eventually notice.

Every point's result is written to disk immediately (not just at the end), and
a per-point exception - including the client subprocess hard-timing-out under
its own backlog - is caught and treated as evidence of saturation rather than
a fatal error. A sweep that dies here still leaves everything it already learned
on disk.

Cold needs roughly 10x the prefill work of Full reuse per request (SPEC section
6), so its knee sits far lower - this is exactly why each workload gets its own
sweep rather than a shared rate grid.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from p1_gates import scrape, GAUGE_CANDIDATES  # noqa: E402 - one source of truth for metric names


def run_point(args, rate: float, run_id: str, available_rows: int) -> dict:
    out_dir = Path(tempfile.mkdtemp(prefix="pilot_"))
    # client.py stops at max(num_requests, min_seconds), not min. Overshooting
    # num_requests above the expected arrival count made an earlier version of
    # this (2.5x headroom) run for 300s+ instead of 120s at low rates: the loop
    # never consults the time check until num_requests is reached, so a
    # too-high count means "wait longer for count", never "stop sooner". The
    # safe direction is a SMALL margin above the Poisson mean (rate*duration) -
    # just enough to survive arrival variance without running out of pre-sliced
    # rows before duration elapses. 1.1x keeps overshoot near 10%, not 5-33x.
    ideal = max(10, int(rate * args.duration * 1.1))
    n = min(ideal, available_rows - args.warmup)
    if ideal > available_rows - args.warmup:
        print(f"  WARNING rate {rate:.2f}: wanted {ideal} requests, only "
              f"{available_rows - args.warmup} available - this point may end "
              f"before {args.duration:.0f}s and understate its true completion "
              f"ratio", file=sys.stderr)
    cmd = [
        args.client_py, "scripts/client.py",
        "--jsonl", args.requests, "--base-url", args.base_url,
        "--model", args.model, "--run-id", run_id,
        "--engine", args.engine, "--workload", args.workload,
        "--rate", str(rate), "--min-seconds", str(args.duration),
        "--num-requests", str(n), "--warmup", str(args.warmup),
        "--out", str(out_dir),
    ]
    if args.image_tokens is not None:
        cmd += ["--image-tokens", str(args.image_tokens)]

    # Pre-scrape: KV/preemption state going INTO this point, so the post-scrape
    # can be read either as an absolute gauge (KV usage right now) or diffed
    # against pre (counters: how many preemptions did just this point cause).
    metrics_pre = scrape(args.base_url)

    # A point already deep in queueing collapse needs real wall-clock time to
    # drain the backlog before the client process can even exit and report a
    # result - this is not stuck, it is what "the queue grows without bound"
    # looks like from the outside. The timeout here exists as a circuit
    # breaker, not a normal-case bound; callers must treat its firing as a
    # saturation signal (see main()), never as "something went wrong".
    timeout_s = args.duration + 180
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        metrics_post = scrape(args.base_url)  # still worth knowing KV state at the crash
        return {"rate": rate, "completion_ratio": None, "ttft_p50_s": None,
                "error": f"client did not return within {timeout_s:.0f}s - "
                         f"treat as saturated, not as a bug",
                "metrics_pre": metrics_pre, "metrics_post": metrics_post}

    metrics_post = scrape(args.base_url)
    # Counters get diffed (how much did THIS point add); gauges are reported
    # as their post-point value only (KV usage right now, not "usage minus
    # usage" which is meaningless for a point-in-time reading).
    metrics_delta = {k: round(v - metrics_pre.get(k, 0), 4)
                     for k, v in metrics_post.items() if k not in GAUGE_CANDIDATES}

    if proc.returncode != 0:
        print(f"  rate {rate:.3f}: client FAILED\n{proc.stderr[-800:]}", file=sys.stderr)
        return {"rate": rate, "completion_ratio": 0.0, "error": proc.stderr[-400:],
                "metrics_pre": metrics_pre, "metrics_post": metrics_post}
    payload = json.loads((out_dir / f"{run_id}.json").read_text())
    s = payload["summary"]
    return {"rate": rate, "completion_ratio": s["completion_ratio"],
            "ttft_p50_s": s["ttft_s"]["p50"], "offered": s["offered"],
            "completed": s["completed"],
            # Within-point knee signal: median TTFT of this point's second half
            # over its first half. >2.0 means the queue grew without bound
            # DURING the point - the faithful reading of SPEC section 6's
            # "open-loop TTFT is a function of run length". This is the same
            # signal that later caught 16 of 28 P3 cells the completion-ratio
            # and blowup triggers had missed; the pilot now stops on it too,
            # so kappa is not calibrated by a laxer rule than the one the
            # matrix is judged by. See learnings/measurement/ttft-growth-signal.md
            "ttft_growth_ratio": s.get("ttft_growth_ratio"),
            "tokens_per_second": s.get("tokens_per_second"),
            "tokens_per_second_per_user": s.get("tokens_per_second_per_user"),
            "isl_tokens_p50": (s.get("isl_tokens") or {}).get("p50"),
            "osl_tokens_p50": (s.get("osl_tokens") or {}).get("p50"),
            "image_tokens_p50": (s.get("image_tokens") or {}).get("p50"),
            "text_tokens_p50": (s.get("text_tokens") or {}).get("p50"),
            "measured_cached_fraction": s.get("measured_cached_fraction"),
            "metrics_pre": metrics_pre, "metrics_post": metrics_post,
            "metrics_delta": metrics_delta}


def save(args, points, kappa, note=""):
    result = {"workload": args.workload, "engine": args.engine,
              "kappa": round(kappa, 3) if kappa is not None else None,
              "note": note, "points": points}
    if kappa is not None:
        result["rate_grid"] = {"0.5k": round(0.5 * kappa, 3),
                               "0.8k": round(0.8 * kappa, 3),
                               "1.2k": round(1.2 * kappa, 3)}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workload", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--requests", default="")
    p.add_argument("--client-py", default="/root/venvs/client/bin/python")
    p.add_argument("--duration", type=float, default=120.0, help="seconds per point")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--start-rate", type=float, default=0.5)
    p.add_argument("--factor", type=float, default=1.8)
    # Built workload files are sized for the matrix's intended top rate (SPEC
    # P0: N = ceil(max_rate * run_seconds) + warmup), not for an open-ended
    # pilot search. Sweeping well past that exhausts the file mid-point and
    # silently truncates the run below its target duration. 12 sits just above
    # the max_rate=8 this repo's workloads were built with; raise only if you
    # also rebuild workloads/ with a higher --max-rate.
    p.add_argument("--max-rate", type=float, default=12.0)
    p.add_argument("--max-points", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--ttft-growth", type=float, default=2.0,
                   help="treat saturated once a single point's within-run "
                        "ttft_growth_ratio (2nd-half median / 1st-half median) "
                        "exceeds this - the primary knee signal, matching the "
                        "P3 retag criterion")
    p.add_argument("--ttft-blowup", type=float, default=8.0,
                   help="secondary signal: saturated once p50 TTFT exceeds "
                        "this multiple of the first point's TTFT. Kept as a "
                        "backstop for the case where a point is uniformly slow "
                        "from its first request (already past knee on arrival), "
                        "which within-run growth alone would not flag")
    p.add_argument("--out", default="")
    p.add_argument("--image-tokens", type=int, default=None,
                   help="defaults to workloads/manifest.json geometry.image_tokens")
    p.add_argument("--manifest", default="workloads/manifest.json")
    args = p.parse_args()

    if not args.requests:
        args.requests = f"workloads/{args.workload}/requests.jsonl"
    if not args.out:
        args.out = f"workloads/{args.workload}/pilot_{args.engine}.json"
    if args.image_tokens is None:
        try:
            args.image_tokens = json.loads(Path(args.manifest).read_text())[
                "geometry"]["image_tokens"]
        except Exception as exc:  # noqa: BLE001 - split is a nice-to-have, not required
            print(f"WARNING: could not read image_tokens from {args.manifest}: "
                  f"{exc} - image/text token split will be unavailable", file=sys.stderr)

    available_rows = sum(1 for _ in open(args.requests))
    points = []
    rate = args.start_rate
    kappa = None
    baseline_ttft = None
    print(f"=== pilot: {args.workload} on {args.engine}, {args.duration:.0f}s/point, "
          f"{available_rows} requests available ===")

    for i in range(args.max_points):
        run_id = f"pilot_{args.workload}_{args.engine}_{i}"
        try:
            r = run_point(args, rate, run_id, available_rows)
        except Exception as exc:  # noqa: BLE001 - a bad point must not lose good ones
            r = {"rate": rate, "completion_ratio": None, "ttft_p50_s": None,
                 "error": f"{type(exc).__name__}: {exc}"}
        points.append(r)
        save(args, points, kappa, note="sweep in progress")  # never lose data again

        ratio, ttft = r.get("completion_ratio"), r.get("ttft_p50_s")
        if ratio is None:
            print(f"  rate {rate:6.2f} req/s  FAILED: {r.get('error')}  [SATURATED]")
            kv = r.get("metrics_post") or {}
            if kv:
                print(f"    KV/preemption state at failure: {kv}")
        else:
            if baseline_ttft is None and ttft:
                baseline_ttft = ttft
            tps = r.get("tokens_per_second")
            isl, osl = r.get("isl_tokens_p50"), r.get("osl_tokens_p50")
            img, txt = r.get("image_tokens_p50"), r.get("text_tokens_p50")
            grow = r.get("ttft_growth_ratio")
            grow_str = f"{grow:.2f}" if grow is not None else "n/a"
            print(f"  rate {rate:6.2f} req/s  completed/offered "
                  f"{r.get('completed', '?')}/{r.get('offered', '?')} = {ratio:.3f}  "
                  f"p50 TTFT {ttft * 1000 if ttft else float('nan'):.0f}ms  "
                  f"growth {grow_str}  "
                  f"TPS {tps if tps is not None else '?'}")
            print(f"    ISL(p50) {isl}  OSL(p50) {osl}  "
                  f"image/text tokens {img}/{txt}  "
                  f"cached_frac {r.get('measured_cached_fraction')}")
            delta = r.get("metrics_delta") or {}
            gauges_now = {k: v for k, v in (r.get("metrics_post") or {}).items()
                         if k in GAUGE_CANDIDATES}
            if delta or gauges_now:
                print(f"    counters this point: {delta}")
                print(f"    gauges (KV usage etc) at end of point: {gauges_now}")

        growth = r.get("ttft_growth_ratio")
        ratio_bad = ratio is None or ratio < args.threshold
        growth_bad = growth is not None and growth > args.ttft_growth
        ttft_bad = (ttft and baseline_ttft
                   and ttft > baseline_ttft * args.ttft_blowup and i > 0)

        def point_is_good(pt):
            """A prior point counts as sub-knee only if it passed ALL three
            signals - the same conjunction used to stop, applied backwards so
            the interpolation anchor is not itself already saturated."""
            if pt is None or pt.get("completion_ratio") is None:
                return False
            if pt["completion_ratio"] < args.threshold:
                return False
            g = pt.get("ttft_growth_ratio")
            if g is not None and g > args.ttft_growth:
                return False
            if (baseline_ttft and pt.get("ttft_p50_s")
                    and pt["ttft_p50_s"] > baseline_ttft * args.ttft_blowup):
                return False
            return True

        if ratio_bad or growth_bad or ttft_bad:
            reason = ("completion ratio" if ratio_bad else
                      "TTFT growth" if growth_bad else "TTFT blowup")
            print(f"  -> saturated ({reason}"
                  f"{f'; growth={growth:.2f}' if growth is not None else ''})")
            prev = points[-2] if len(points) > 1 else None
            good = point_is_good(prev)
            if good and growth_bad and prev.get("ttft_growth_ratio") is not None:
                # Growth was the trigger and completion_ratio is typically 1.0
                # on both sides here, so interpolate on growth crossing the
                # threshold, not on ratio (which would degenerate to a blind
                # midpoint). Linear between the last sub-threshold growth and
                # this over-threshold one.
                r0, g0 = prev["rate"], prev["ttft_growth_ratio"]
                r1, g1 = r["rate"], growth
                frac = (args.ttft_growth - g0) / (g1 - g0) if g1 != g0 else 0.5
                frac = min(max(frac, 0.0), 1.0)
                kappa = r0 + frac * (r1 - r0)
            elif good and ratio is not None:
                # Completion-ratio (or blowup) trigger: interpolate on ratio
                # crossing --threshold between the last good point and this one.
                r0, c0 = prev["rate"], prev["completion_ratio"]
                r1, c1 = r["rate"], ratio
                frac = (c0 - args.threshold) / (c0 - c1) if c0 != c1 else 0.5
                kappa = r0 + frac * (r1 - r0)
            elif good:
                # This point errored out entirely (e.g. subprocess timeout) with
                # no ratio to interpolate against - the last good rate is itself
                # the safest usable estimate, since whatever lies between it and
                # this failure is already in collapse.
                kappa = prev["rate"]
            else:
                kappa = rate  # saturated on the very first point tried
            break
        rate *= args.factor
        if rate > args.max_rate:
            kappa = rate / args.factor
            print(f"  reached max-rate without saturating; using last good rate")
            break
    else:
        kappa = points[-1]["rate"]

    result = save(args, points, kappa, note="complete")
    print(f"\nkappa ~= {kappa:.3f} req/s" if kappa is not None else "\nkappa: undetermined")
    if kappa is not None:
        print(f"rate grid: {result['rate_grid']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
