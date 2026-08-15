#!/usr/bin/env python3
"""Aggregate the clean re-run (D4c) + graph experiments (D2) into the corrected
comparison that feeds P4. Run after the re-run pulls back to workloads/*/results.

    python3 scripts/analyze_rerun.py

Consumes, if present:
  workloads/{full-reuse,partial-reuse}/results/*_{low,mid}_rep{1,2,3}.json
  workloads/cold/results/cold_*_mid_rep{1,2,3}.json
  workloads/{full-reuse,partial-reuse}/results/*_sglang_{stock,forced}_rep*.json
  workloads/{full-reuse,partial-reuse}/pilot_sglang.json + pilot_sglang_graphson.json

Prints per-cell n=3 stats (p50/p95/p99 with spread), the cross-engine gap at
each clean rate, the graph-A/B delta, and the stock-vs-graphs-on knee. Everything
degrades gracefully: sections whose inputs are missing are skipped with a note,
so it is safe to run against a partially-collected matrix.
"""

import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RATES = {  # for labelling; must match scripts/rerun_clean.sh
    "full-reuse": {"low": 0.94, "mid": 1.50},
    "partial-reuse": {"low": 0.60, "mid": 0.90},
    "cold": {"mid": 0.48},
}


def pctl(sorted_vals, q):
    """Nearest-rank percentile (q in 0..100) over an already-sorted list, ms."""
    if not sorted_vals:
        return None
    import math
    k = max(1, math.ceil(q / 100 * len(sorted_vals)))
    return sorted_vals[k - 1]


def load(pattern):
    """Return (summary, measured_ttfts_ms) per matching result file. Percentiles
    are computed from the raw per-request measured TTFTs, not the summary's fixed
    p50/p90/p99 set, so any percentile (e.g. cold's p90) is available and correct
    over completers only."""
    out = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        s = d.get("summary")
        if not s:
            continue
        warmup = s.get("warmup", 0)
        ttfts = sorted(r["ttft_s"] * 1000 for r in d.get("requests", [])
                       if r.get("index", 0) >= warmup and r.get("error") is None
                       and r.get("ttft_s") is not None)
        out.append((s, ttfts))
    return out


def agg(rows, pct="p99"):
    """n-run mean and spread for p50 and the chosen tail percentile, computed
    from raw per-request TTFTs. pct is like 'p90'/'p95'/'p99'."""
    if not rows:
        return None
    q = int(pct[1:])
    p50 = [pctl(t, 50) for _, t in rows if t]
    tail = [pctl(t, q) for _, t in rows if t]
    growth = [s.get("ttft_growth_ratio") for s, _ in rows
              if s.get("ttft_growth_ratio") is not None]
    tags = sorted({t for s, _ in rows for t in s.get("tags", [])})
    return {
        "n": len(rows),
        "p50_mean": statistics.mean(p50) if p50 else None,
        "p50_spread": (max(p50) - min(p50)) if p50 else None,
        "tail_pct": pct,
        "tail_mean": statistics.mean(tail) if tail else None,
        "completed_min": min(s["completed"] for s, _ in rows),
        "growth_max": max(growth) if growth else None,
        "tags": tags,
    }


def fmt(a):
    if a is None:
        return "  (no data yet)"
    p50 = f"{a['p50_mean']:.0f}" if a["p50_mean"] is not None else "n/a"
    spread = f"{a['p50_spread']:.0f}" if a["p50_spread"] is not None else "?"
    tail = f"{a['tail_mean']:.0f}" if a["tail_mean"] is not None else "n/a"
    return (f"  n={a['n']}  p50={p50}ms (spread {spread}) "
            f"{a['tail_pct']}={tail}ms  completed>={a['completed_min']}  "
            f"growth_max={a['growth_max']}  tags={a['tags']}")


def section_clean_matrix():
    print("=" * 72)
    print("CLEAN RE-RUN (D4c) — per engine per clean rate, n=3")
    print("=" * 72)
    for w in ("full-reuse", "partial-reuse", "cold"):
        # Cold: p90 (well-powered at n=300 and emitted by the harness); p99 for
        # cold is underpowered by design (its low arrival rate makes ~1000
        # completions cost ~25 min/cell). Cache workloads: p99 (n=500).
        pct = "p90" if w == "cold" else "p99"
        print(f"\n## {w}   (tail = {pct}{' , cache-off baseline' if w=='cold' else ''})")
        for band in RATES[w]:
            rate = RATES[w][band]
            print(f"\n  rate {band} = {rate} req/s")
            gaps = {}
            for engine in ("vllm", "sglang"):
                s = load(f"workloads/{w}/results/{w}_{engine}_{band}_rep*.json")
                a = agg(s, pct)
                print(f"   {engine:6s}{fmt(a)}")
                if a:
                    gaps[engine] = a["p50_mean"]
            if "vllm" in gaps and "sglang" in gaps and gaps["vllm"]:
                g = (gaps["sglang"] - gaps["vllm"]) / gaps["vllm"] * 100
                print(f"   -> engine gap (SGLang vs vLLM, p50): {g:+.1f}%")


def section_graph_ab():
    print("\n" + "=" * 72)
    print("GRAPH A/B (D2) — SGLang stock vs forced prefill graphs, at clean rate")
    print("=" * 72)
    any_data = False
    for w in ("full-reuse", "partial-reuse"):
        stock = agg(load(f"workloads/{w}/results/{w}_sglang_stock_rep*.json"), "p99")
        forced = agg(load(f"workloads/{w}/results/{w}_sglang_forced_rep*.json"), "p99")
        if not stock and not forced:
            continue
        any_data = True
        print(f"\n## {w}")
        print(f"   stock {fmt(stock)}")
        print(f"   forced{fmt(forced)}")
        if stock and forced and stock["p50_mean"]:
            d = (forced["p50_mean"] - stock["p50_mean"]) / stock["p50_mean"] * 100
            print(f"   -> forced-vs-stock p50: {d:+.1f}%  "
                  f"({'graphs help' if d < 0 else 'graphs do not help'})")
        # capture outcome
        for arm in ("stock", "forced"):
            gl = ROOT / f"gates/sglang-graph-ab/{arm}_{w}_graph_lines.txt"
            if gl.exists() and gl.read_text().strip():
                print(f"   [{arm}] capture log: {gl.read_text().strip().splitlines()[-1][:110]}")
    if not any_data:
        print("  (no graph-A/B data yet)")


def section_graph_knee():
    print("\n" + "=" * 72)
    print("GRAPHS-ON KNEE PILOT (D2) — does forcing prefill graphs move SGLang's knee?")
    print("=" * 72)
    any_data = False
    for w in ("full-reuse", "partial-reuse"):
        def kappa(fn):
            p = ROOT / f"workloads/{w}/{fn}"
            if p.exists():
                try:
                    return json.load(open(p)).get("kappa")
                except json.JSONDecodeError:
                    return None
            return None
        stock_k, graph_k = kappa("pilot_sglang.json"), kappa("pilot_sglang_graphson.json")
        if stock_k is None and graph_k is None:
            continue
        any_data = True
        vk = kappa("pilot_vllm.json")
        line = f"\n## {w}: SGLang stock κ={stock_k}  graphs-on κ={graph_k}"
        if vk:
            line += f"  (vLLM κ={vk})"
        print(line)
        if stock_k and graph_k:
            print(f"   -> knee change: {(graph_k-stock_k)/stock_k*100:+.0f}%  "
                  f"({'graphs lift the knee' if graph_k>stock_k*1.15 else 'knee ~unchanged -> not graph-bound'})")
    if not any_data:
        print("  (no graphs-on knee pilot yet)")


if __name__ == "__main__":
    section_clean_matrix()
    section_graph_ab()
    section_graph_knee()
    print()
