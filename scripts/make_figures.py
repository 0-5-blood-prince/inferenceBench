#!/usr/bin/env python3
"""P4: render figures/ttft.png and figures/gap.png from the extracted run table.

Run scripts/extract_p4.py first (writes scripts/_p4_data.json).

Colour: dataviz reference palette, categorical slots 1 and 2, used unchanged.
Engine identity is *also* carried by marker shape and line style, so the figures
survive greyscale printing and CVD without relying on hue.

Fill convention, both figures: solid marker = that run passed both saturation
checks; hollow marker = tagged `saturated`. A comparison is only legitimate
where *both* engines are solid.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"
ENG = {
    "vllm":   {"c": "#2a78d6", "m": "o", "ls": "-",  "label": "vLLM 0.26.0"},
    "sglang": {"c": "#eb6834", "m": "s", "ls": "--", "label": "SGLang 0.5.16"},
}
PANELS = [("full-reuse", "Full reuse"), ("cold", "Cold (cache off)"),
          ("partial-reuse", "Partial reuse")]
KAPPA = ["0.5κ", "0.8κ", "1.2κ"]

ROWS = [r for r in json.load(open("scripts/_p4_data.json"))]


def series(workload, engine):
    """(rate, mean p50, min, max, saturated) per rate tag, in rate order."""
    out = {}
    for r in ROWS:
        if r["workload"] != workload or r["engine"] != engine:
            continue
        out.setdefault(r["rate"], []).append(r)
    pts = []
    for rate in sorted(out):
        reps = out[rate]
        p50 = [x["ttft_p50"] for x in reps]
        pts.append((rate, sum(p50) / len(p50), min(p50), max(p50),
                    any(x["saturated"] for x in reps), len(reps)))
    return pts


def fig_ttft(path="figures/ttft.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (wl, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        rates = sorted({p[0] for e in ENG for p in series(wl, e)})

        # Shade the only rate where both engines stayed sub-knee.
        ax.axvspan(rates[0] * 0.93, rates[0] * 1.07, color="#1baf7a", alpha=0.10,
                   lw=0, zorder=0)

        for eng, st in ENG.items():
            pts = series(wl, eng)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, st["ls"], color=st["c"], lw=2, zorder=3,
                    label=st["label"], alpha=0.9)
            for x, y, lo, hi, sat, n in pts:
                if n > 1:  # min-max range across replicates
                    ax.plot([x, x], [lo, hi], color=st["c"], lw=1.2, zorder=4,
                            solid_capstyle="butt")
                ax.plot([x], [y], marker=st["m"], ms=9, zorder=5,
                        color=st["c"] if not sat else SURFACE,
                        markerfacecolor=st["c"] if not sat else SURFACE,
                        markeredgecolor=st["c"], markeredgewidth=2)

        ax.set_yscale("log")
        ax.set_xticks(rates)
        ax.set_xticklabels([f"{r:.2f}\n{k}" for r, k in zip(rates, KAPPA)],
                           fontsize=9, color=INK2)
        ax.set_xlim(min(rates) * 0.78, max(rates) * 1.12)
        ax.set_title(title, fontsize=12, color=INK, pad=9)
        ax.set_xlabel("offered rate (req/s)", fontsize=10, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)

    axes[0].set_ylabel("p50 TTFT (seconds) — LOG SCALE", fontsize=10.5,
                       color=INK)
    axes[0].annotate("only rate where\nboth engines are\nsub-saturation",
                     xy=(0.055, 0.955), xycoords="axes fraction", fontsize=8.5,
                     color="#0f7a55", va="top", ha="left", linespacing=1.35)

    handles = [Line2D([], [], color=st["c"], marker=st["m"], ls=st["ls"], lw=2,
                      ms=9, label=st["label"]) for st in ENG.values()]
    handles += [
        Line2D([], [], color=INK2, marker="o", ls="none", ms=9,
               markerfacecolor=INK2, label="sub-saturation (usable)"),
        Line2D([], [], color=INK2, marker="o", ls="none", ms=9,
               markerfacecolor=SURFACE, markeredgewidth=2,
               label="saturated (queueing, not a point estimate)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.045), labelcolor=INK2)

    fig.suptitle("Time-to-first-token vs offered load — Gemma-4-31B, A100 80GB",
                 fontsize=13.5, color=INK, y=1.0)
    fig.text(0.5, 0.925,
             "Vertical bars are min–max across replicates (n=3 on Full reuse "
             "0.8κ/1.2κ; n=1 elsewhere). Axis is log-scaled: the plotted range "
             "spans 0.23 s to 96 s.",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {path}")


def fig_gap(path="figures/gap.png"):
    h7 = json.load(open("scripts/_p4_h7.json"))
    pts = sorted(h7["points"])  # (cached fraction %, gap %)
    labels = {0.0: "Cold\n0.63 req/s",
              53.7: "Partial reuse\n1.13 req/s",
              86.0: "Full reuse\n2.04 req/s"}

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "--", color=INK2, lw=1.4, alpha=0.55, zorder=2)
    ax.plot(xs, ys, "o", ms=12, color="#2a78d6", markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=4)

    # Measured labels go ABOVE their point; predictions go BELOW their diamond.
    # Keeping the two families on opposite sides is what prevents collisions.
    for x, y in pts:
        key = min(labels, key=lambda k: abs(k - x))
        ax.annotate(f"{labels[key]}\ngap {y:.1f}%", xy=(x, y),
                    xytext=(4 if x < 10 else 0, 16), textcoords="offset points",
                    ha="left" if x < 10 else "center", fontsize=9.5, color=INK,
                    linespacing=1.4)

    # Staggered vertically so neither label sits on a drop line or the
    # interpolation segment.
    for frac, key, name, dy in [(40.0, "pred_40", "Conversation", -16),
                                (59.0, "pred_59", "Tool&Agent", -36)]:
        g = h7[key]
        ax.plot([frac, frac], [46.5, g], ":", color="#eb6834", lw=1.5, zorder=3)
        ax.plot([frac], [g], marker="D", ms=9, color=SURFACE,
                markeredgecolor="#eb6834", markeredgewidth=2, zorder=5)
        ax.annotate(f"{name}  {frac:.0f}% → {g:.1f}%", xy=(frac, g),
                    xytext=(-11, dy), textcoords="offset points", ha="right",
                    fontsize=9, color="#b8471a", linespacing=1.35)

    ax.text(0.055, 0.20, "3 POINTS,\nNOT A FITTED TREND",
            transform=ax.transAxes, ha="left", va="center", fontsize=13,
            color="#b8471a", fontweight="bold", linespacing=1.4)

    ax.set_xlabel("measured cached-token fraction (%)  —  mean of the two "
                  "engines", fontsize=10.5, color=INK, labelpad=8)
    ax.set_ylabel("SGLang p50 TTFT penalty vs vLLM (%)", fontsize=10.5, color=INK)
    ax.set_title("Engine TTFT gap vs prefix reuse — sub-saturation runs only "
                 "(0.5κ)", fontsize=12.5, color=INK, pad=12)
    ax.set_xlim(-8, 100)
    ax.set_ylim(44, 84)
    ax.grid(color=GRID, lw=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)

    handles = [
        Line2D([], [], color="#2a78d6", marker="o", ls="none", ms=11,
               label="measured (both engines sub-saturation)"),
        Line2D([], [], color="#eb6834", marker="D", ls="none", ms=9,
               markerfacecolor=SURFACE, markeredgewidth=2,
               label="H7 prediction (interpolated, unobserved)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9.5,
              labelcolor=INK2)

    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.text(0.5, 0.085,
             "Dashed line is piecewise-linear interpolation between adjacent "
             "points, drawn only to read off the H7 predictions.",
             ha="center", va="top", fontsize=8.8, color=INK2)
    fig.text(0.5, 0.048,
             "Cached fraction is confounded with offered rate — both rise "
             "together across these three workloads, so this is not a clean "
             "single-variable curve.",
             ha="center", va="top", fontsize=8.8, color=INK2)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    fig_ttft()
    fig_gap()
