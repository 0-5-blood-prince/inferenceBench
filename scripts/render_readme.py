#!/usr/bin/env python3
"""Regenerate a workload's README from what is on disk (SPEC section 8).

    python3 scripts/render_readme.py full-reuse
    python3 scripts/render_readme.py --all

Run once at P0 exit so the setup diagram is reviewable while the results table
is still empty, then after every cell in P3. The README is derived output and is
never hand-edited: anything the generator cannot read off disk belongs in the
top-level README instead.
"""

import argparse
import json
import re
from pathlib import Path

DESCRIPTIONS = {
    "full-reuse": (
        "Every request opens identically: the same system prompt, the same "
        "image, and the same block of text. Only a short unique tail at the end "
        "differs, so roughly seven eighths of each request is a repeat of the "
        "one before it. This is the best case a cache can ever have, and it "
        "answers a single question: when reuse fully works, does either engine "
        "pull ahead?"
    ),
    "cold": (
        "Every request gets its own freshly generated noise image and its own "
        "text, and caching is switched off at the server flag as well. Nothing "
        "repeats and nothing can be skipped, so both engines do all the work "
        "every time. This is the control. If the engines differ here, the "
        "difference is not about caching at all, and every other result in the "
        "matrix has to be read in that light."
    ),
    "partial-reuse": (
        "Every request opens with the same image and the same canonical text, "
        "but each one stops copying at a different, randomly chosen point and "
        "then goes its own way. One request might share 300 tokens and the next "
        "780. This is the sharp probe: one engine tracks its cache in fixed "
        "16-token chunks and has to discard a chunk a request diverges inside, "
        "the other tracks it token by token and does not. Randomising where "
        "requests split, rather than only what they say, is what stops both "
        "engines from getting conveniently aligned split points."
    ),
    "single-stream": (
        "Full-reuse content, but one request at a time and longer answers. "
        "There is no queue, no batching and nothing to schedule, so the GPU is "
        "barely occupied. This isolates plain text-generation speed away from "
        "every caching question - the check that both engines generate at the "
        "same rate when nothing clever is happening."
    ),
    "visionarena-replay": (
        "Real recorded traffic from people chatting with vision models: real "
        "photos, real questions, real variation in length. The sampled "
        "conversations are unrelated to each other, so almost nothing is shared "
        "between requests. That makes this the cold case with genuine content - "
        "a check that the no-reuse result survives real inputs instead of "
        "synthetic noise."
    ),
    "mooncake-replay": (
        "Production traces from a real deployment, replayed with their original "
        "arrival timing - bursty, not smoothed - and with markers showing which "
        "requests genuinely shared openings. Two traces with known sharing "
        "rates of roughly 40% and 59%. The point is the ordering: predictions "
        "for these two numbers are read off the synthetic results and committed "
        "before this workload is ever run, so the goalposts cannot move."
    ),
}

# Counter names differ per engine and their exact semantics are pinned at Gate B
# (P1). Anything not found here renders as "n/a" rather than a guess.
# vLLM exposes only aggregate counters, so its cached fraction is a counter diff
# across the run. SGLang's sglang:cache_hit_rate is a GAUGE and reads 0.0 after a
# run even when reuse demonstrably happened - never diff it. SGLang's per-request
# cached_tokens (via --enable-cache-report) is captured by the client instead and
# surfaces as summary.measured_cached_fraction.
HIT_COUNTERS = {
    "vllm": ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"),
}


def read_counter(text: str, name: str):
    match = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$",
                      text, re.MULTILINE)
    return float(match.group(1)) if match else None


def cached_fraction(folder: Path, run_id: str, engine: str):
    """Measured cached-token fraction across one run, from the pre/post scrapes."""
    pair = HIT_COUNTERS.get((engine or "").lower())
    if not pair:
        return None
    pre_path = folder / "metrics" / f"{run_id}_pre.txt"
    post_path = folder / "metrics" / f"{run_id}_post.txt"
    if not (pre_path.exists() and post_path.exists()):
        return None
    pre, post = pre_path.read_text(), post_path.read_text()
    hits, queries = pair
    dh = (read_counter(post, hits) or 0) - (read_counter(pre, hits) or 0)
    dq = (read_counter(post, queries) or 0) - (read_counter(pre, queries) or 0)
    return dh / dq if dq > 0 else None


def load_runs(folder: Path):
    runs = []
    for path in sorted((folder / "results").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        summary = payload.get("summary", payload)
        summary["_run_id"] = summary.get("run_id", path.stem)
        runs.append(summary)
    runs.sort(key=lambda s: (s.get("engine") or "", s.get("rate") or 0))
    return runs


def ms(value):
    return f"{value * 1000:.0f}" if isinstance(value, (int, float)) else "—"


def pct(value):
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "n/a"


def diagram(slug: str, manifest: dict) -> str:
    geom = manifest["geometry"]
    info = manifest.get("workloads", {}).get(slug, {})
    total = geom["T"]
    sys_tok, img_tok = geom["system_tokens"], geom["image_tokens"]
    canonical, tail = geom["canonical_tokens"], geom["full_reuse_tail_tokens"]

    if slug == "cold":
        blocks = [
            ("S", f"system prompt<br/>{sys_tok} tok", "shared"),
            ("I", f"unique image<br/>{img_tok} tok", "uniq"),
            ("U", f"unique text<br/>{total - sys_tok - img_tok} tok", "uniq"),
        ]
    elif slug == "partial-reuse":
        blocks = [
            ("S", f"system prompt<br/>{sys_tok} tok", "shared"),
            ("I", f"shared image<br/>{img_tok} tok", "shared"),
            ("C", f"canonical prefix<br/>E = {geom['E_min']}..{geom['E_max']} tok"
                  "<br/>varies per request", "shared"),
            ("F", "unique filler<br/>absorbs the change<br/>so the total is fixed",
             "uniq"),
            ("Q", f"question<br/>{geom['question_tokens']} tok", "uniq"),
        ]
    else:  # full-reuse, single-stream
        blocks = [
            ("S", f"system prompt<br/>{sys_tok} tok", "shared"),
            ("I", f"shared image<br/>{img_tok} tok", "shared"),
            ("C", f"canonical text<br/>{canonical} tok", "shared"),
            ("T", f"unique tail<br/>{tail} tok", "uniq"),
        ]

    lines = [
        "```mermaid",
        "flowchart LR",
        f'    subgraph REQ["one request — {total} input tokens, always"]',
        "        direction LR",
    ]
    for key, label, cls in blocks:
        lines.append(f'        {key}["{label}"]:::{cls}')
    lines.append("        " + " --> ".join(k for k, _, _ in blocks))
    lines.append("    end")

    served = []
    if info.get("cache_disabled"):
        served.append("prefix caching DISABLED by flag")
    if info.get("concurrency") == 1:
        served.append("concurrency 1, no queue")
    served.append(f"max_tokens {info.get('output_budget', '?')}, ignore_eos")
    lines += [
        f'    CFG["served under:<br/>{"<br/>".join(served)}"]',
        "    REQ --> CFG",
        "    classDef shared fill:#d7ebd7,stroke:#4a7c4a,color:#173d17",
        "    classDef uniq fill:#f5dcdc,stroke:#a35050,color:#4a1414",
        "```",
        "",
        "Green segments are byte-identical across every request in this "
        "workload; red segments are unique to each request.",
    ]
    return "\n".join(lines)


def validity_rule(slug: str, manifest: dict):
    geom = manifest["geometry"]
    if slug == "full-reuse":
        target = geom["full_reuse_cacheable_fraction"]
        return (f"Measured cached-token fraction must sit within ±5 points of the "
                f"constructed {target * 100:.1f}% — not near 100%, the unique tail "
                f"is real.",
                lambda f: f is not None and abs(f - target) <= 0.05)
    if slug == "cold":
        return ("Measured cached-token fraction must read approximately 0.",
                lambda f: f is not None and f <= 0.05)
    if slug == "partial-reuse":
        lo, hi = manifest["partial_reuse_observed_fraction_span"]
        return (f"Measured cached-token fraction must land inside the constructed "
                f"span {lo * 100:.1f}%–{hi * 100:.1f}%.",
                lambda f: f is not None and lo - 0.05 <= f <= hi + 0.05)
    return ("No cache-fraction gate applies to this workload; completion rate is "
            "the only validity condition.", lambda f: True)


def render(slug: str, root: Path) -> Path:
    folder = root / slug
    manifest = json.loads((root / "manifest.json").read_text())
    name = manifest.get("workloads", {}).get(slug, {}).get("name", slug)
    runs = load_runs(folder)
    rule_text, rule = validity_rule(slug, manifest)

    out = [
        f"# {name}",
        "",
        "<!-- Generated by scripts/render_readme.py — do not edit by hand. -->",
        "",
        DESCRIPTIONS.get(slug, ""),
        "",
        "## Setup",
        "",
        diagram(slug, manifest),
        "",
    ]

    geom = manifest["geometry"]
    out += [
        "| Constant | Value |",
        "|---|---|",
        f"| Input budget `T` | {geom['T']} tokens, identical across workloads |",
        f"| Image resolution | {manifest['resolution']} |",
        f"| Requests built | {manifest['requests_per_workload']} "
        f"({manifest['warmup']} discarded as warmup) |",
        f"| Tokenizer | `{manifest['tokenizer']}` |",
        "",
        "## Results",
        "",
    ]

    if not runs:
        out += ["_No runs yet._ This README was rendered before the matrix; the "
                "table fills in as each cell completes.", ""]
    else:
        out += [
            "| Run | Engine | Rate (req/s) | p50 TTFT (ms) | p99 TTFT (ms) | "
            "p50 ITL (ms) | Completed / offered | Cached fraction | Tag |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for run in runs:
            frac = cached_fraction(folder, run["_run_id"], run.get("engine", ""))
            ttft = run.get("ttft_s") or {}
            itl = run.get("itl_s") or {}
            rate = run.get("rate")
            out.append(
                f"| `{run['_run_id']}` | {run.get('engine') or '—'} | "
                f"{'—' if run.get('concurrency') == 1 else rate} | "
                f"{ms(ttft.get('p50'))} | {ms(ttft.get('p99'))} | "
                f"{ms(itl.get('p50'))} | "
                f"{run.get('completed', '—')} / {run.get('offered', '—')} | "
                f"{pct(frac)} | {', '.join(run.get('tags') or []) or '—'} |")
        out.append("")
        out.append("p50 is primary. At 200 requests p99 is the second-worst "
                   "sample and is descriptive only.")
        out.append("")

    out += ["## Validity", "", rule_text, ""]
    if runs:
        out += ["| Run | Completion ≥ 95% | Cache gate | Verdict |",
                "|---|---|---|---|"]
        for run in runs:
            frac = cached_fraction(folder, run["_run_id"], run.get("engine", ""))
            completed_ok = (run.get("completion_ratio") or 0) >= 0.95
            gate_ok = rule(frac)
            if frac is None and slug != "single-stream":
                verdict, gate = "not yet checkable", "no scrape"
            elif not gate_ok:
                verdict, gate = "**void** — workload broken, latencies unusable", "fail"
            elif not completed_ok:
                verdict, gate = "saturated — excluded from sub-knee claims", "pass"
            else:
                verdict, gate = "ok", "pass"
            out.append(f"| `{run['_run_id']}` | {'yes' if completed_ok else 'no'} | "
                       f"{gate} | {verdict} |")
        out.append("")

    out += ["---", "",
            "Design, hypotheses and thresholds: [`spec/SPEC.md`](../../spec/SPEC.md). "
            "Cross-workload figures and the H1–H7 verdicts live in the top-level "
            "README."]

    path = folder / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workload", nargs="?", help="folder name, e.g. partial-reuse")
    p.add_argument("--all", action="store_true", help="render every built workload")
    p.add_argument("--root", default="workloads")
    args = p.parse_args()

    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    known = list(manifest.get("workloads", {}))

    if args.all:
        targets = known
    elif args.workload:
        targets = [args.workload]
    else:
        raise SystemExit(f"name a workload or pass --all. built: {', '.join(known)}")

    for slug in targets:
        if slug not in known:
            raise SystemExit(f"unknown workload {slug!r}; built: {', '.join(known)}")
        print(render(slug, root))


if __name__ == "__main__":
    main()
