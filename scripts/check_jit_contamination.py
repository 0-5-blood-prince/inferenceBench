#!/usr/bin/env python3
"""Per-run validity check: did a JIT/graph-capture event fire during MEASUREMENT?

    check_jit_contamination.py --log logs/vllm-warm.log --since-line 4021 \
        --result workloads/full-reuse/results/<run_id>.json

Confirmed real on this pod, not hypothetical: vLLM's own log emits
"Triton kernel JIT compilation during inference: kernel_unified_attention. This
causes a latency spike; consider extending warmup to cover this shape/config"
- and it appeared inside measured windows across 10 separate run logs before
this check existed. A 20-request warmup does not reliably touch every
batch-size bucket or tensor shape that triggers a distinct kernel compile or
capture; whichever cell happens to hit one gets a latency spike injected into
its TTFT/ITL numbers, and the existing validity check (completion ratio,
cached-fraction band) has no way to see it - a JIT-contaminated run silently
passes as `ok`.

This does not try to eliminate the event (raising warmup to 50 requests helps
but cannot guarantee every shape is covered); it detects it after the fact and
taggs the run so it can be excluded from pooled analysis rather than silently
polluting it, the same way `saturated` and `budget_unpinned` already are.

Usage: call once per cell, after the client run completes, with --since-line
set to the log's line count captured immediately BEFORE that cell started (so
only lines appended during THIS run's warmup+measurement are scanned). Mutates
the result JSON's summary.tags in place if contamination is found.
"""

import argparse
import calendar
import json
import re
import sys
import time
from pathlib import Path

PATTERN = re.compile(r"JIT compilation during inference|jit_monitor.*WARNING",
                     re.IGNORECASE)
# vLLM/SGLang log lines carry a "MM-DD HH:MM:SS" timestamp in the container's
# clock (UTC on these pods). Parse it to an epoch so a JIT event can be placed
# before or after the measured window began.
TS = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\b")


def line_epoch(line: str, year: int) -> float | None:
    m = TS.search(line)
    if not m:
        return None
    mo, da, hh, mm, ss = (int(x) for x in m.groups())
    try:
        # UTC on these pods (Etc/UTC); calendar.timegm treats the tuple as UTC.
        return calendar.timegm((year, mo, da, hh, mm, ss, 0, 0, 0))
    except (ValueError, OverflowError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="engine log file for this cell")
    p.add_argument("--since-line", type=int, required=True,
                   help="line count of --log immediately before this cell started")
    p.add_argument("--result", required=True, help="results/<run_id>.json to tag")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"WARNING: {args.log} not found, cannot check JIT contamination",
              file=sys.stderr)
        sys.exit(0)  # do not block the run on a missing log

    lines = log_path.read_text(errors="replace").splitlines()
    new_lines = lines[args.since_line:]
    all_hits = [l for l in new_lines if PATTERN.search(l)]

    result_path = Path(args.result)
    payload = json.loads(result_path.read_text())
    summary = payload["summary"]

    # A JIT compile absorbed by WARMUP is the intended, harmless case for these
    # fixed-shape workloads (the compile fires once on the first request of a
    # shape and never again). Only a compile that fires INSIDE the measured
    # window injects a latency spike into the reported percentiles. Use the
    # client-recorded measured-window start (summary.measured_start_epoch) to
    # keep only the hits at/after it; without that field (older runs), fall back
    # to the original behaviour of flagging any hit in the slice.
    measured_start = summary.get("measured_start_epoch")
    if measured_start is not None:
        year = time.gmtime(measured_start).tm_year
        # -2s guard: the log line is written a moment after the event; do not let
        # sub-second/rounding place a genuine measured-window event just before.
        cutoff = measured_start - 2
        hits = [l for l in all_hits
                if (line_epoch(l, year) is None or line_epoch(l, year) >= cutoff)]
        warmup_only = len(all_hits) - len(hits)
        if warmup_only:
            print(f"note: {warmup_only} JIT event(s) fell in warmup (ignored) in "
                  f"{args.result}")
    else:
        hits = all_hits

    if hits:
        summary.setdefault("tags", [])
        if "jit_contaminated" not in summary["tags"]:
            summary["tags"].append("jit_contaminated")
        summary["jit_contamination_lines"] = hits[:5]  # a sample, not the whole dump
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"JIT CONTAMINATION: {len(hits)} event(s) during measurement in "
              f"{args.result} - tagged jit_contaminated")
        print(f"  first: {hits[0][:200]}")
        sys.exit(1)

    print(f"clean: no JIT/capture events during measurement in {args.result}")


if __name__ == "__main__":
    main()
