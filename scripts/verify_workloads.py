#!/usr/bin/env python3
"""Verify the built workloads satisfy the SPEC section 5 invariants.

Run at P0 exit, before any server time is spent. A workload that violates these
does not produce noisy data - it produces void data, and the SPEC section 6
hit-rate gate will only reveal it after the runs are already burned.

    python3 scripts/verify_workloads.py --root workloads

Checks, per workload:
  - total input token count is exactly T, for every request (structure varies,
    volume never does)
  - Full reuse, Partial reuse and Single stream share one image; Cold's images
    are all distinct
  - the canonical block is token-identical across every Full-reuse request
  - each Partial-reuse request's shared region is a true token prefix of that
    same canonical block, not merely a string that looks like one
  - Partial-reuse divergence points spread across residues mod the vLLM block
    size
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def user_text(row) -> str:
    return row["messages"][1]["content"][1]["text"]


def image_uri(row) -> str:
    return row["messages"][1]["content"][0]["image_url"]["url"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="workloads")
    p.add_argument("--tokenizer", help="defaults to the manifest's tokenizer")
    p.add_argument("--block-size", type=int, default=16,
                   help="vLLM block size, for the alignment spread check")
    args = p.parse_args()

    from transformers import AutoTokenizer

    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    geom = manifest["geometry"]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or manifest["tokenizer"])

    def ids(text: str):
        return tok.encode(text, add_special_tokens=False)

    total, head = geom["T"], geom["shared_head_tokens"]
    image_tokens = geom["image_tokens"]
    failures = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' - {detail}' if detail else ''}")
        if not ok:
            failures.append(label)

    info = manifest["workloads"]
    sets = {slug: load(root / slug / "requests.jsonl") for slug in info}

    for slug, rows in sets.items():
        print(f"\n{info[slug]['name']}  ({len(rows)} requests, {slug}/)")
        counts = {
            len(ids(r["messages"][0]["content"])) + image_tokens + len(ids(user_text(r)))
            for r in rows
        }
        check(counts == {total}, f"every request is exactly T={total} input tokens",
              f"observed {sorted(counts)}")
        budget = info[slug]["output_budget"]
        check(all(r["max_tokens"] == budget and r["ignore_eos"] for r in rows),
              f"output budget pinned at {budget} with ignore_eos")
        check((root / slug / "results").is_dir() and (root / slug / "metrics").is_dir(),
              "results/ and metrics/ folders exist")

    uniq = {slug: len({image_uri(r) for r in rows}) for slug, rows in sets.items()}
    print("\nimage sharing")
    for slug in ("full-reuse", "partial-reuse", "single-stream"):
        check(uniq[slug] == 1, f"{info[slug]['name']} uses one shared image")
    check(uniq["cold"] == len(sets["cold"]), "Cold images are all unique",
          f"{uniq['cold']} distinct of {len(sets['cold'])}")
    shared = {image_uri(sets[s][0]) for s in ("full-reuse", "partial-reuse",
                                              "single-stream")}
    check(len(shared) == 1,
          "the shared image is byte-identical across all three workloads that use it")

    print("\nprefix sharing")
    canonical_n = geom["canonical_tokens"]
    canonical = ids(user_text(sets["full-reuse"][0]))[:canonical_n] if canonical_n else []
    if canonical_n:
        check(all(ids(user_text(r))[:canonical_n] == canonical
                  for r in sets["full-reuse"]),
              "Full-reuse canonical block is token-identical across requests")
        check(len({user_text(r)[-40:] for r in sets["full-reuse"]}) > 1,
              "Full-reuse tails differ per request")

    bad = [r["request_id"] for r in sets["partial-reuse"]
           if ids(user_text(r))[:r["constructed_cacheable_tokens"] - head]
           != canonical[:r["constructed_cacheable_tokens"] - head]]
    check(not bad,
          "Partial-reuse shared regions are true token prefixes of that same block",
          f"{len(bad)} violations" if bad else "")

    offsets = Counter((r["constructed_cacheable_tokens"] - head) % args.block_size
                      for r in sets["partial-reuse"])
    check(len(offsets) > args.block_size // 2,
          f"Partial-reuse divergence points spread across residues mod "
          f"{args.block_size}",
          f"{len(offsets)}/{args.block_size} residues hit")

    fracs = [r["constructed_cacheable_fraction"] for r in sets["partial-reuse"]]
    check(min(fracs) <= 0.35 and max(fracs) >= 0.75,
          "Partial-reuse cacheable fraction spans roughly 30-80 percent",
          f"{min(fracs):.3f} - {max(fracs):.3f}")
    check(0.85 <= geom["full_reuse_cacheable_fraction"] <= 0.90,
          "Full-reuse constructed cacheable fraction inside the SPEC section 6 band",
          f"{geom['full_reuse_cacheable_fraction']:.3f}")

    if failures:
        print(f"\n{len(failures)} FAILED - workloads are not usable")
        sys.exit(1)
    print("\nall invariants hold")


if __name__ == "__main__":
    main()
