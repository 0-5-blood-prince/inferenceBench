#!/usr/bin/env python3
"""P4: turn the extracted run table into the numbers the README asserts.

Everything here is derived from scripts/_p4_data.json (written by
extract_p4.py). Nothing is typed in by hand.

Gap convention throughout: gap% = (SGLang_p50 - vLLM_p50) / vLLM_p50 * 100.
Positive = SGLang slower. This is fixed once here so the README, the figures
and the H7 prediction cannot drift apart.
"""
import json
import statistics as st

ROWS = json.load(open("scripts/_p4_data.json"))
CORE = ["full-reuse", "cold", "partial-reuse"]
RATE_LABEL = {0.5: "0.5k", 0.8: "0.8k", 1.2: "1.2k"}


def cells(workload, rate_tag):
    """All reps for a workload/engine/rate, keyed by engine."""
    out = {}
    for r in ROWS:
        if r["workload"] != workload:
            continue
        if rate_tag and rate_tag not in r["run_id"].split("_", 2)[-1]:
            continue
        out.setdefault(r["engine"], []).append(r)
    return out


def agg(reps):
    p50s = [r["ttft_p50"] for r in reps]
    return {
        "n": len(reps),
        "p50_mean": st.mean(p50s),
        "p50_min": min(p50s),
        "p50_max": max(p50s),
        "spread_pct": (max(p50s) - min(p50s)) / min(p50s) * 100 if len(p50s) > 1 else 0.0,
        "itl": st.mean([r["itl_p50"] for r in reps]),
        "tok_s": st.mean([r["tok_s"] for r in reps]),
        "sat": any(r["saturated"] for r in reps),
        "growth_max": max(r["growth"] for r in reps),
        # cached fraction: counter-derived (works for both engines); fall back
        # to the per-request field where the counter is absent (SGLang cold has
        # no cached_tokens_total series at all when radix cache is disabled).
        "frac": next((r["counter_frac"] for r in reps if r["counter_frac"] is not None),
                     next((r["per_req_frac"] for r in reps
                           if r["per_req_frac"] is not None), None)),
        "rate": reps[0]["rate"],
    }


def gap(v, s):
    return (s - v) / v * 100


def line(w, tag, label=None):
    c = cells(w, tag)
    if "vllm" not in c or "sglang" not in c:
        return None
    v, s = agg(c["vllm"]), agg(c["sglang"])
    return {
        "workload": w, "rate_tag": label or tag, "rate": v["rate"],
        "v": v, "s": s, "gap": gap(v["p50_mean"], s["p50_mean"]),
        "clean": not v["sat"] and not s["sat"],
    }


print("=" * 100)
print("CORE MATRIX -- p50 TTFT (s), gap%, cached fraction, clean?")
print("=" * 100)
print(f"{'workload':<15}{'rate':<7}{'req/s':>7}{'vLLM':>9}{'SGL':>9}{'gap%':>8}"
      f"{'v.frac':>8}{'s.frac':>8}{'n_v':>4}{'n_s':>4}  clean")
core_lines = []
for w in CORE:
    for tag in ["0.5k", "0.8k", "1.2k"]:
        L = line(w, tag)
        if not L:
            continue
        core_lines.append(L)
        vf = f"{L['v']['frac']:.3f}" if L["v"]["frac"] is not None else "off"
        sf = f"{L['s']['frac']:.3f}" if L["s"]["frac"] is not None else "off"
        print(f"{w:<15}{tag:<7}{L['rate']:>7.3f}{L['v']['p50_mean']:>9.3f}"
              f"{L['s']['p50_mean']:>9.3f}{L['gap']:>8.1f}{vf:>8}{sf:>8}"
              f"{L['v']['n']:>4}{L['s']['n']:>4}  {'YES' if L['clean'] else 'no'}")

L = line("single-stream", "")
core_lines.append(L)
print(f"{'single-stream':<15}{'conc=1':<7}{'-':>7}{L['v']['p50_mean']:>9.3f}"
      f"{L['s']['p50_mean']:>9.3f}{L['gap']:>8.1f}{L['v']['frac']:>8.3f}"
      f"{L['s']['frac']:>8.3f}{L['v']['n']:>4}{L['s']['n']:>4}  YES")
print(f"  single-stream ITL p50: vLLM {L['v']['itl']:.4f}s  SGLang {L['s']['itl']:.4f}s"
      f"  -> delta {gap(L['v']['itl'], L['s']['itl']):+.2f}%  [H5]")

print()
print("=" * 100)
print("TEXT-ONLY CONTROL vs MULTIMODAL  (all at each workload's own 0.5k)")
print("=" * 100)
print(f"{'workload':<16}{'vLLM ms':>9}{'SGL ms':>9}{'gap%':>8}   |  "
      f"{'text vLLM':>10}{'text SGL':>10}{'gap%':>8}{'shrink pts':>12}")
for w in ["full-reuse", "partial-reuse"]:
    mm = line(w, "0.5k")
    tx = line(w + "-text", "0.5k")
    shrink = mm["gap"] - tx["gap"]
    print(f"{w:<16}{mm['v']['p50_mean']*1000:>9.1f}{mm['s']['p50_mean']*1000:>9.1f}"
          f"{mm['gap']:>8.1f}   |  {tx['v']['p50_mean']*1000:>10.1f}"
          f"{tx['s']['p50_mean']*1000:>10.1f}{tx['gap']:>8.1f}{shrink:>12.1f}")
    print(f"{'':16}  n={mm['v']['n']}/{mm['s']['n']}"
          f"{'':>28} n={tx['v']['n']}/{tx['s']['n']}"
          f"   ({shrink/mm['gap']*100:.0f}% of the gap is multimodal-specific)")

print()
print("=" * 100)
print("VARIANCE BLOCK -- Full reuse, n=3, why 0.8k is not a comparison point")
print("=" * 100)
for tag in ["0.8k", "1.2k"]:
    c = cells("full-reuse", tag)
    for eng in ["vllm", "sglang"]:
        a = agg(c[eng])
        reps = "/".join(f"{r['ttft_p50']:.2f}" for r in sorted(c[eng], key=lambda x: x["run_id"]))
        print(f"  {tag} {eng:<7} n={a['n']}  p50 = {reps} s   spread {a['spread_pct']:.1f}%"
              f"   growth {a['growth_max']:.2f}  {'SATURATED' if a['sat'] else 'ok'}")

print()
print("=" * 100)
print("GAP.PNG POINTS -- both engines sub-knee only")
print("=" * 100)
pts = [L for L in core_lines if L["clean"] and L["workload"] in CORE]
for L in pts:
    vfrac = L["v"]["frac"] or 0.0
    sfrac = L["s"]["frac"] or 0.0
    mean_frac = (vfrac + sfrac) / 2
    print(f"  {L['workload']:<15} rate={L['rate']:.3f} req/s  cached: vLLM {vfrac*100:.1f}% "
          f"SGLang {sfrac*100:.1f}%  mean {mean_frac*100:.1f}%   gap {L['gap']:.1f}%")
print(f"  -> {len(pts)} points. Absolute offered rate varies {min(L['rate'] for L in pts):.2f}"
      f"-{max(L['rate'] for L in pts):.2f} req/s across them (cached fraction is")
print("     confounded with load; both move together across workloads).")


def interp(x, pts_xy):
    pts_xy = sorted(pts_xy)
    for (x0, y0), (x1, y1) in zip(pts_xy, pts_xy[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


xy = [(((L["v"]["frac"] or 0.0) + (L["s"]["frac"] or 0.0)) / 2 * 100, L["gap"]) for L in pts]
print()
print("=" * 100)
print("H7 PREDICTIONS -- piecewise-linear interpolation between adjacent points")
print("=" * 100)
for name, frac in [("Mooncake Conversation", 40.0), ("Mooncake Tool&Agent", 59.0)]:
    print(f"  {name:<24} cached={frac:.0f}%  ->  predicted gap {interp(frac, xy):.1f}%")
c40, c59 = interp(40.0, xy), interp(59.0, xy)
print(f"  Ordering check (H7 clause 2, 'Tool&Agent gap >= Conversation gap'):")
print(f"    Tool&Agent {c59:.1f}% vs Conversation {c40:.1f}%  ->  "
      f"{'HOLDS' if c59 >= c40 else 'ALREADY CONTRADICTED by the day-1 curve'}")

json.dump({"points": xy, "pred_40": c40, "pred_59": c59},
          open("scripts/_p4_h7.json", "w"), indent=1)
