#!/usr/bin/env python3
"""Build the synthetic workloads (SPEC section 5).

Run once, in P0, after the per-image token count has been probed. Everything
downstream depends on the constants this script computes and records; the
manifest it writes is what P2 freezes.

    python3 workloads/build.py \
        --image-tokens 256 --resolution 896x896 \
        --tokenizer google/gemma-3-27b-it \
        --max-rate 16 --run-seconds 240

Output layout (SPEC section 8) - one self-contained folder per workload:

    workloads/images/                 shared + unique noise images
    workloads/manifest.json           frozen constants
    workloads/full-reuse/requests.jsonl
    workloads/full-reuse/{results,metrics}/
    workloads/cold/...                and so on

Geometry
--------
Let S = system prompt tokens, I = per-image tokens, T = the common input budget.

    Full reuse     [SYS][IMG shared][canonical block C][tail]   cacheable (S+I+C)/T
    Cold           [SYS][IMG unique][unique text]               cacheable ~0 (flag)
    Partial reuse  [SYS][IMG shared][canonical prefix E][filler][Q]
                                                                cacheable (S+I+E)/T
    Single stream  Full-reuse content, concurrency 1, longer outputs

Full reuse carries the entire canonical block, making it the E=C endpoint of the
Partial-reuse sweep (SPEC section 5). That is not cosmetic: Partial reuse's
cacheable fraction bottoms out at (S+I)/T, the same quantity Full reuse pins at
85-90%, so no single T can put that ratio at both 85% and 30%. Sharing the block
resolves it and makes the two workloads one continuous axis.

--full-reuse-canonical=exclude builds the unsatisfiable variant on purpose, as a
diagnostic: it fails the band assertion and prints the arithmetic.
"""

import argparse
import base64
import hashlib
import io
import json
import random
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a careful visual analysis assistant. Examine the provided image and "
    "answer the user's question using only what is visible. Be concise and "
    "concrete. Do not speculate about anything outside the frame."
)

QUESTION = "Given all of the above, describe what you see in the image."

# Folder name -> display name. These slugs are the on-disk contract: run ids,
# results, metrics and the generated README all live under them.
WORKLOADS = {
    "full-reuse": "Full reuse",
    "cold": "Cold",
    "partial-reuse": "Partial reuse",
    "single-stream": "Single stream",
}

# Ordinary words, space separated, so token slices land on word boundaries and
# re-encode stably. Content is irrelevant; only token counts and sharing matter.
VOCAB = (
    "time person year way day thing man world life hand part child eye woman "
    "place work week case point government company number group problem fact be "
    "have do say get make go know take see come think look want give use find "
    "tell ask work seem feel try leave call good new first last long great "
    "little own other old right big high different small large next early young "
    "important few public bad same able the of and a in that it for not on with "
    "as you at this but his by from they we her she or an will my one all would "
    "there their what so up out about who which when make can like just him know "
    "into your over think also back after use two how our well even want because "
    "any these give most us water light sound field study story month night book "
    "line road river house school system program question during number without "
    "before between under against through above below across behind beyond"
).split()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TextMaker:
    """Generates text with exact token counts under a specific tokenizer."""

    def __init__(self, tokenizer, seed: int):
        self.tok = tokenizer
        self.rng = random.Random(seed)

    def encode(self, text: str):
        return self.tok.encode(text, add_special_tokens=False)

    def _blob(self, n_words: int) -> str:
        return " ".join(self.rng.choice(VOCAB) for _ in range(n_words))

    def exact(self, n_tokens: int) -> str:
        """Text that encodes to exactly n_tokens tokens."""
        if n_tokens <= 0:
            return ""
        text = self._blob(max(8, int(n_tokens * 1.5)))
        while len(self.encode(text)) < n_tokens + 16:
            text += " " + self._blob(64)
        for _ in range(24):
            ids = self.encode(text)
            if len(ids) == n_tokens:
                return text
            if len(ids) > n_tokens:
                text = self.tok.decode(ids[:n_tokens]).rstrip()
            else:
                text += " " + self._blob(max(1, n_tokens - len(ids)))
        raise RuntimeError(f"could not hit exactly {n_tokens} tokens")

    def word_boundary_prefixes(self, text: str):
        """Token-count cut points of `text` that fall on word boundaries.

        A prefix cut here re-encodes to the identical token ids as the full
        text's leading tokens, so every Partial-reuse request shares a genuine
        token prefix rather than one that merely looks shared as a string.
        """
        ids = self.encode(text)
        cuts = []
        for n in range(1, len(ids) + 1):
            piece = self.tok.decode(ids[:n])
            if piece and piece[-1].isspace():
                continue
            if self.encode(piece) == ids[:n] and (
                n == len(ids) or self.tok.decode(ids[n : n + 1]).startswith((" ", "\n"))
            ):
                cuts.append((n, piece))
        return cuts


def noise_image(width: int, height: int, seed: int) -> bytes:
    """Deterministic RGB noise.

    numpy, not random.getrandbits per byte: at 448x448 that is 600k Python-level
    calls per image, and the matrix needs thousands of images. Seeded, so the
    images remain reproducible from the manifest.
    """
    import numpy as np
    from PIL import Image

    raw = np.random.default_rng(seed).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(raw, mode="RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def compute_geometry(args, s_tokens: int, q_tokens: int) -> dict:
    """Solve for T, the canonical block length C, and Partial reuse's E range."""
    head = s_tokens + args.image_tokens  # the always-shared [SYS][IMG] region

    # T is pinned by Partial reuse's lower band: at E = E_min the cacheable
    # fraction must come down to frac_min.
    total = int(-(-head // args.partial_frac_min))  # ceil
    tail = max(1, round(total * (1.0 - args.full_reuse_frac)))

    if args.full_reuse_canonical == "include":
        canonical = total - head - tail
    else:
        canonical = 0
        tail = total - head

    e_min = max(0, round(args.partial_frac_min * total) - head)
    e_max = int(args.partial_frac_max * total) - head
    if args.full_reuse_canonical == "include":
        e_max = min(e_max, canonical)

    full_frac = (head + canonical) / total
    geom = {
        "T": total,
        "system_tokens": s_tokens,
        "image_tokens": args.image_tokens,
        "shared_head_tokens": head,
        "canonical_tokens": canonical,
        "full_reuse_tail_tokens": tail,
        "question_tokens": q_tokens,
        "E_min": e_min,
        "E_max": e_max,
        "full_reuse_cacheable_fraction": round(full_frac, 4),
        "partial_reuse_cacheable_fraction_min": round((head + e_min) / total, 4),
        "partial_reuse_cacheable_fraction_max": round((head + e_max) / total, 4),
    }

    problems = []
    if not 0.85 <= full_frac <= 0.90:
        problems.append(
            f"Full-reuse cacheable fraction {full_frac:.3f} outside SPEC section 6 "
            f"band [0.85, 0.90]"
        )
    if e_max <= e_min:
        problems.append(
            f"Partial reuse has no room to vary: E_min={e_min} E_max={e_max}")
    if total - head - e_max - q_tokens < 0:
        problems.append("Partial-reuse filler would be negative at E_max")
    if problems:
        detail = "\n  ".join(problems)
        hint = ""
        if args.full_reuse_canonical == "exclude":
            hint = (
                "\n\nExpected: without the canonical block, Full reuse's cacheable "
                "fraction is (S+I)/T, which is also Partial reuse's minimum. One T "
                "cannot put that ratio at 0.85+ and 0.30 at once. This is why "
                "SPEC section 5 gives Full reuse the whole block. "
                "Use --full-reuse-canonical=include."
            )
        sys.exit(f"geometry does not satisfy SPEC:\n  {detail}{hint}")
    return geom


def build(args) -> None:
    from transformers import AutoTokenizer

    out = Path(args.out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for slug in WORKLOADS:
        for sub in ("results", "metrics"):
            (out / slug / sub).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tm = TextMaker(tok, args.seed)

    s_tokens = len(tm.encode(SYSTEM_PROMPT))
    q_tokens = len(tm.encode(QUESTION))
    geom = compute_geometry(args, s_tokens, q_tokens)
    total, canonical_n = geom["T"], geom["canonical_tokens"]

    width, height = (int(v) for v in args.resolution.lower().split("x"))

    # One request stream, reused by every rate and both engines: N covers the
    # longest run at the highest rate, so no run ever recycles a "unique" image.
    n_requests = max(args.num_requests, int(-(-args.max_rate * args.run_seconds // 1)))
    n_total = n_requests + args.warmup

    # One copy of the shared image on disk, referenced by all three workloads
    # that need it - identity by construction, not by assertion.
    shared_png = noise_image(width, height, seed=args.seed)
    (img_dir / "shared.png").write_bytes(shared_png)
    shared_uri = data_uri(shared_png)

    unique_uris = []
    for i in range(n_total):
        png = noise_image(width, height, seed=args.seed + 1 + i)
        (img_dir / f"unique_{i:05d}.png").write_bytes(png)
        unique_uris.append(data_uri(png))

    canonical_text = tm.exact(canonical_n) if canonical_n else ""
    cuts = tm.word_boundary_prefixes(canonical_text) if canonical_text else []
    if canonical_text and not cuts:
        sys.exit("canonical block produced no word-boundary cut points")

    rng = random.Random(args.seed + 9999)

    def message(image_uri: str, text: str) -> list:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": text},
                ],
            },
        ]

    def record(idx, messages, max_tokens, cacheable_tokens):
        return {
            "request_id": idx,
            "messages": messages,
            "max_tokens": max_tokens,
            # min_tokens + empty stop_token_ids alongside ignore_eos: ignore_eos
            # covers only the tokenizer's eos_token_id, not the chat template's
            # extra stop tokens, and an early stop turns into a fake latency win.
            "min_tokens": max_tokens,
            "ignore_eos": True,
            "stop_token_ids": [],
            "temperature": 0.0,
            "constructed_cacheable_tokens": cacheable_tokens,
            "constructed_cacheable_fraction": round(cacheable_tokens / total, 4),
        }

    head = geom["shared_head_tokens"]
    rows = {slug: [] for slug in WORKLOADS}

    for i in range(n_total):
        tail = tm.exact(geom["full_reuse_tail_tokens"])
        shared_text = (canonical_text + " " + tail).strip()

        rows["full-reuse"].append(
            record(i, message(shared_uri, shared_text),
                   args.max_tokens, head + canonical_n))
        # Cold keeps T identical but shares nothing beyond the system prompt,
        # and the cache is disabled by engine flag on top of that.
        rows["cold"].append(
            record(i, message(unique_uris[i], tm.exact(total - head)),
                   args.max_tokens, 0))
        rows["single-stream"].append(
            record(i, message(shared_uri, shared_text),
                   args.single_stream_max_tokens, head + canonical_n))

        # Partial reuse: the divergence point moves per request. Randomizing the
        # offset, not just the content, is what denies both engines a stable
        # block alignment; filler absorbs the change so T never moves.
        e_target = rng.randint(geom["E_min"], geom["E_max"])
        e_actual, prefix = 0, ""
        for n, piece in cuts:
            if n <= e_target:
                e_actual, prefix = n, piece
            else:
                break
        filler = tm.exact(total - head - e_actual - q_tokens)
        text = " ".join(x for x in (prefix, filler, QUESTION) if x)
        rows["partial-reuse"].append(
            record(i, message(shared_uri, text), args.max_tokens, head + e_actual))

    for slug, records in rows.items():
        with (out / slug / "requests.jsonl").open("w") as fh:
            for row in records:
                fh.write(json.dumps(row) + "\n")

    fracs = [r["constructed_cacheable_fraction"] for r in rows["partial-reuse"]]
    manifest = {
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "resolution": args.resolution,
        "full_reuse_canonical": args.full_reuse_canonical,
        "geometry": geom,
        "requests_per_workload": n_total,
        "warmup": args.warmup,
        "max_rate": args.max_rate,
        "run_seconds": args.run_seconds,
        "workloads": {
            slug: {
                "name": name,
                "requests": f"{slug}/requests.jsonl",
                "output_budget": (args.single_stream_max_tokens
                                  if slug == "single-stream" else args.max_tokens),
                "concurrency": 1 if slug == "single-stream" else None,
                "cache_disabled": slug == "cold",
                "sha256": sha256_file(out / slug / "requests.jsonl"),
            }
            for slug, name in WORKLOADS.items()
        },
        "partial_reuse_observed_fraction_span": [min(fracs), max(fracs)],
        "shared_image_sha256": hashlib.sha256(shared_png).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(geom, indent=2))
    print(f"\nPartial-reuse constructed fraction span: "
          f"{min(fracs):.3f} - {max(fracs):.3f}")
    print(f"{n_total} requests per workload")
    for slug in WORKLOADS:
        print(f"  {out / slug}")
    print(f"images: {img_dir}")
    print("\nNext: render each workload README before any run (SPEC section 8):")
    print(f"  for w in {' '.join(WORKLOADS)}; do "
          f"python3 scripts/render_readme.py $w; done")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-tokens", type=int, required=True,
                   help="per-image token count from the P0 probe")
    p.add_argument("--resolution", required=True, help="fixed image resolution, e.g. 896x896")
    p.add_argument("--tokenizer", required=True, help="HF model id for token counting")
    p.add_argument("--out-dir", default="workloads")
    p.add_argument("--max-rate", type=float, required=True,
                   help="highest offered rate in the matrix, req/s")
    p.add_argument("--run-seconds", type=int, default=240)
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--single-stream-max-tokens", type=int, default=512)
    p.add_argument("--full-reuse-frac", type=float, default=0.875)
    p.add_argument("--partial-frac-min", type=float, default=0.30)
    p.add_argument("--partial-frac-max", type=float, default=0.80)
    p.add_argument("--full-reuse-canonical", choices=("include", "exclude"),
                   default="include",
                   help="include: Full reuse shares the whole canonical block, per "
                        "SPEC section 5. exclude: diagnostic only, fails the bands.")
    build(p.parse_args())


if __name__ == "__main__":
    main()
