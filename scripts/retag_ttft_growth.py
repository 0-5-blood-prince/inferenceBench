#!/usr/bin/env python3
"""One-time retag: apply the TTFT-growth saturation signal to already-collected
P3 results, without re-running anything on the GPU.

Fixes a real gap found reviewing the actual P3 matrix output: half of the 28
real cells showed TTFT growing to 10-180 SECONDS (SPEC section 6's own stated
definition of saturation - "the queue grows without bound") while still tagged
`ok`, because the original tagging only checked completion ratio, which stays
near 1.0 even during severe queueing collapse (the engine still finishes every
request, just far too slowly). The raw per-request data is entirely valid;
only the tag was wrong. This recomputes tags in place using the same
ttft_growth_ratio() now built into client.py's live summarize(), so no cell
needs to be re-run - see client.py's ttft_growth_ratio() docstring for why the
threshold (>2.0) was chosen and verified against this exact data first.

    python3 scripts/retag_ttft_growth.py workloads/*/results/*.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from client import ttft_growth_ratio  # noqa: E402


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: retag_ttft_growth.py <result.json> [...]")

    changed = []
    for p in paths:
        path = Path(p)
        payload = json.loads(path.read_text())
        summary = payload["summary"]
        ok = [r for r in payload["requests"]
              if r["error"] is None and r["ttft_s"] is not None]
        ratio = ttft_growth_ratio(ok)
        summary["ttft_growth_ratio"] = ratio

        was_saturated = "saturated" in summary.get("tags", [])
        ratio_saturated = ratio is not None and ratio > 2.0
        completion_saturated = summary.get("completion_ratio", 1.0) < 0.95
        now_saturated = completion_saturated or ratio_saturated

        tags = [t for t in summary.get("tags", []) if t not in ("ok", "saturated")]
        summary["tags"] = (["saturated"] if now_saturated else ["ok"]) + tags

        if now_saturated != was_saturated:
            changed.append((path.name, ratio, was_saturated, now_saturated))
        path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Processed {len(paths)} files, {len(changed)} tag changed:")
    for name, ratio, before, after in changed:
        print(f"  {name:40s} ratio={ratio:6.2f}  saturated: {before} -> {after}")


if __name__ == "__main__":
    main()
