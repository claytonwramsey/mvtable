#!/usr/bin/env python3
"""Scatter CAPT/MVT (SIMD) motion-planning time against primitive-only time, per MBM problem
instance (robot, dataset, scene_id), pooled across all selected robots into one plot.

Reads `data/mbm_plan_results.csv` (produced by `cargo run --release -p mbm-plan-bench`) and writes
an SVG (default `doc/primitive_vs_other.svg`, + a `.png` sibling). Only instances solved under
*both* the primitive baseline and the structure being compared are plotted: an unsolved row is
capped at the planner's time budget, not a real timing measurement, so mixing it in would read as
a data point when it's actually a timeout artifact.

Mirrors rumple's `scripts/plot_primitive_vs_other.py` (Okabe-Ito colorblind-safe series colors,
log-log axes, despined Tufte range-frame axes), swapping that script's mesh/point-cloud-vs-primitive
comparison for this repo's own point-cloud structures (CAPT, MVT) against the same primitive-only
baseline:

    python3 scripts/plot_primitive_vs_other.py
    python3 scripts/plot_primitive_vs_other.py --structures mvtable_simd --robots panda ur5
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mbm_common import (
    STRUCTURE_COLORS,
    YLABEL_PAD,
    save_figure,
    style_legend,
    trim_spines_to_data,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "mbm_plan_results.csv"
DEFAULT_OUT = ROOT / "doc" / "primitive_vs_other.svg"

BASELINE = "primitive"

# Every SIMD-capable point-cloud structure in `mbm_plan_results.csv`. Every series here is
# SIMD-only with no non-SIMD sibling in the same chart to set apart with a lightened color (as
# `plot_mbm.py` does for its query-time panel), so this uses the bold `STRUCTURE_COLORS` directly -
# a lightened pastel is hard to tell apart from another lightened pastel once both are further
# faded by the scatter's own alpha blending.
ALL_STRUCTURES = ["capt_simd", "mvtable_simd", "mvtable_mutable_simd", "mvt_cpp_simd"]
DEFAULT_STRUCTURES = ["capt_simd", "mvtable_simd"]
COLORS = {
    "capt_simd": STRUCTURE_COLORS["capt"],
    "mvtable_simd": STRUCTURE_COLORS["mvtable"],
    "mvtable_mutable_simd": STRUCTURE_COLORS["mvtable_mutable"],
    "mvt_cpp_simd": STRUCTURE_COLORS["mvt_cpp"],
}
LABELS = {
    "capt_simd": "CAPT",
    "mvtable_simd": "MVT",
    "mvtable_mutable_simd": "Mutable MVT (SIMD)",
    "mvt_cpp_simd": "MVT (C++, SIMD)",
}
REFERENCE_COLOR = "#1A1A1A"
ROBOT_ORDER = ["panda", "ur5", "fetch", "baxter"]

# Fraction of the combined log-span to pad the view beyond the real data range on each side, so a
# fit line terminates with visible clearance before the axes edge and no scatter point sits flush
# against a spine, instead of both looking clipped. Applied in log space (not raw data units) so
# it reads as constant visual spacing regardless of a panel's actual order-of-magnitude span.
VIEW_MARGIN_FRAC = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=DEFAULT_DATA,
        help=f"input mbm-plan-bench results CSV (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help=f"output SVG path; a same-named .png sibling is also written (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        choices=ROBOT_ORDER,
        default=None,
        help="subset of robots to pool into the plot (default: all robots present in the data)",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        choices=ALL_STRUCTURES,
        default=DEFAULT_STRUCTURES,
        help="subset of SIMD structures to plot against the primitive-only baseline "
        f"(default: {' '.join(DEFAULT_STRUCTURES)})",
    )
    return parser.parse_args()


def _draw_scatter_panel(
    ax, groups: list[tuple[str, tuple, pd.Series, pd.Series]]
) -> bool:
    """Draw the primitive-vs-structure scatter into `ax` from `groups`, a list of `(label, color,
    x, y)` scatter series (already filtered to non-empty). Returns False (and leaves `ax` blank)
    if `groups` is empty."""
    if not groups:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax)
        return False

    # The axes should span exactly the scatter points actually plotted, not the fit lines or the
    # equal-time reference line below - those can run past the real data's range (e.g. a fit line
    # extrapolating slightly beyond its narrowest input) without dragging the trimmed axes out
    # with them.
    x_lo = min(x.min() for _, _, x, _ in groups)
    x_hi = max(x.max() for _, _, x, _ in groups)
    y_lo = min(y.min() for _, _, _, y in groups)
    y_hi = max(y.max() for _, _, _, y in groups)

    # Span the reference line across the full combined range so it still reads as a diagonal even
    # when the x and y ranges differ - matplotlib clips it to whichever axis view ends up
    # narrower below, rather than this needing to compute that itself.
    ref_lo, ref_hi = min(x_lo, y_lo), max(x_hi, y_hi)

    def _padded_view(lo: float, hi: float) -> tuple[float, float]:
        """Pad `[lo, hi]` by `VIEW_MARGIN_FRAC` in log space, so the plot box is a little larger
        than the tightest data extent on that axis."""
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        pad = VIEW_MARGIN_FRAC * (log_hi - log_lo)
        return 10 ** (log_lo - pad), 10 ** (log_hi + pad)

    ax.plot(
        [ref_lo, ref_hi],
        [ref_lo, ref_hi],
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1,
        zorder=1,
        label="Equal time",
    )

    for label, color, x, y in groups:
        ax.scatter(
            x,
            y,
            s=10,
            alpha=0.35,
            color=color,
            edgecolors="none",
            label=label,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Each axis views its own margin-padded data range rather than a shared, symmetric one - x and
    # y rarely span the same number of decades, and forcing them to match pads whichever axis is
    # narrower well past its real data, leaving two dead triangular corners in the (aspect-equal)
    # plot box. Letting each axis hug its own range instead keeps the box tight; `set_aspect`
    # below still shrinks the physical box to the narrower dimension so a decade in x and a decade
    # in y cover the same physical distance, which is what makes "45 degrees" mean equal time.
    ax.set_xlim(*_padded_view(x_lo, x_hi))
    ax.set_ylim(*_padded_view(y_lo, y_hi))
    ax.set_aspect("equal", adjustable="box")

    sns.despine(ax=ax)
    # Tufte range-frame: trim spines to the actual scatter data extent (not the fit/reference
    # lines' possibly-wider reach) and label the exact min/max instead of only the nearest
    # round-number tick. Also gives the log axes plain decimal tick labels ("100", not "10^2").
    trim_spines_to_data(ax, xlim=(x_lo, x_hi), ylim=(y_lo, y_hi))

    handles, labels = ax.get_legend_handles_labels()
    # The equal-time reference line is a guide, not a data series, so it reads last rather than
    # wherever it happened to be drawn.
    order = sorted(range(len(labels)), key=lambda i: labels[i] == "Equal time")
    # The lower-right corner of the view sits below the equal-time diagonal, where CAPT/MVT are
    # consistently slower than the primitive baseline and the scatter is sparse - an inset legend
    # there doesn't cover any real data.
    style_legend(
        ax,
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower right",
    )
    return True


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    solved = df[df.solved]
    key = ["robot", "dataset", "scene_id"]
    pivot = solved.pivot_table(
        index=key, columns="structure", values="time_secs", aggfunc="first"
    )
    # The data may not include every structure (e.g. a filtered or partial run) - reindex so
    # "primitive" and any requested structure are always present as columns, filled with NaN
    # rather than raising a KeyError below.
    needed_cols = [BASELINE, *args.structures]
    pivot = pivot.reindex(columns=needed_cols)
    pivot[needed_cols] *= 1000  # seconds -> milliseconds

    available_robots = set(pivot.index.get_level_values("robot"))
    requested_robots = args.robots if args.robots is not None else ROBOT_ORDER
    robots = [r for r in ROBOT_ORDER if r in requested_robots]
    missing_robots = [r for r in robots if r not in available_robots]
    for r in missing_robots:
        print(f"warning: no data for robot {r!r}, skipping")
    robots = [r for r in robots if r in available_robots]

    if not robots:
        print("no data to plot for the requested robots/structures")
        return

    combined = pivot.loc[robots]
    pairs = {
        name: combined[[BASELINE, name]].dropna()
        for name in args.structures
        if name in combined.columns
    }
    pairs = {name: pair for name, pair in pairs.items() if not pair.empty}
    for name in args.structures:
        n = len(pairs.get(name, []))
        print(f"{name}: {n} instances solved under both primitive and {name}")

    groups = [
        (LABELS[name], COLORS[name], pair[BASELINE], pair[name])
        for name, pair in pairs.items()
    ]

    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot()
    _draw_scatter_panel(ax, groups)

    ax.set_xlabel("Planning time with primitives (ms)")
    if len(args.structures) == 1:
        y_label = f"{LABELS[args.structures[0]]} time (ms)"
    else:
        y_label = "Planning time with point cloud (ms)"
    ax.set_ylabel(y_label, labelpad=-8)

    fig.tight_layout()
    save_figure(fig, args.output, crop=True, crop_png=True)


if __name__ == "__main__":
    main()
