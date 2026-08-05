#!/usr/bin/env python3
"""Scatter CAPT/MVT (SIMD) motion-planning time against primitive-only time, per MBM problem
instance (robot, dataset, scene_id), faceted by robot (or pooled into one joint plot with
`--joint`).

Reads `data/mbm_plan_results.csv` (produced by `cargo run --release -p mbm-plan-bench`) and writes
an SVG (default `doc/primitive_vs_other.svg`, + a `.png` sibling). Only instances solved under
*both* the primitive baseline and the structure being compared are plotted: an unsolved row is
capped at the planner's time budget, not a real timing measurement, so mixing it in would read as
a data point when it's actually a timeout artifact.

Mirrors rumple's `scripts/plot_primitive_vs_other.py` (Okabe-Ito colorblind-safe series colors,
log-log axes, despined Tufte range-frame axes, one shared legend below the whole figure), swapping
that script's mesh/point-cloud-vs-primitive comparison for this repo's own point-cloud
structures (CAPT, MVT) against the same primitive-only baseline:

    python3 scripts/plot_primitive_vs_other.py
    python3 scripts/plot_primitive_vs_other.py --structures mvtable_simd --robots panda ur5
    python3 scripts/plot_primitive_vs_other.py --joint
"""

import argparse
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mbm_common import (
    ROBOT_LABELS,
    STRUCTURE_COLORS,
    save_figure,
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
    "capt_simd": "CAPT (SIMD)",
    "mvtable_simd": "MVT (SIMD)",
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
        help="subset of robots to plot, one panel each (default: all robots present in the data)",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        choices=ALL_STRUCTURES,
        default=DEFAULT_STRUCTURES,
        help="subset of SIMD structures to plot against the primitive-only baseline "
        f"(default: {' '.join(DEFAULT_STRUCTURES)})",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="pool all selected robots into one plot instead of faceting into one panel per robot",
    )
    return parser.parse_args()


def _draw_scatter_panel(
    ax, groups: list[tuple[str, tuple, pd.Series, pd.Series]], title: str
) -> bool:
    """Draw one primitive-vs-structure scatter panel into `ax` from `groups`, a list of
    `(label, color, x, y)` scatter series (already filtered to non-empty). Shared by the per-robot
    faceted panels and the pooled `--joint` panel - the only difference between those two callers
    is how many series they hand in and what color/label each one gets. Returns False (and leaves
    `ax` blank) if `groups` is empty."""
    if not groups:
        ax.set_title(title, fontsize=10)
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
    # when the x and y ranges differ, rather than clipping it to one axis's narrower span.
    ref_lo, ref_hi = min(x_lo, y_lo), max(x_hi, y_hi)

    # Pad that combined range by a fixed fraction in log space for the *view* (see
    # VIEW_MARGIN_FRAC), so the plot box is a little larger than the tightest data/line extent.
    log_lo, log_hi = np.log10(ref_lo), np.log10(ref_hi)
    pad = VIEW_MARGIN_FRAC * (log_hi - log_lo)
    view_lo, view_hi = 10 ** (log_lo - pad), 10 ** (log_hi + pad)

    ax.plot(
        [view_lo, view_hi],
        [view_lo, view_hi],
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
    # View limits span the combined, margin-padded, symmetric range rather than each axis's own
    # narrower data range, so the two corners of the plot box both land exactly on the equal-time
    # line - it wouldn't hit the corners if x and y kept their own tighter, unequal ranges. The
    # spines themselves still trim to the real per-axis data below, so this padding shows up as
    # visible spacing between a spine's end (or a fit line's terminus) and the plot edge, rather
    # than stretching the spine/tick labels to match.
    ax.set_xlim(view_lo, view_hi)
    ax.set_ylim(view_lo, view_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)

    sns.despine(ax=ax)
    # Tufte range-frame: trim spines to the actual scatter data extent (not the fit/reference
    # lines' possibly-wider reach) and label the exact min/max instead of only the nearest
    # round-number tick.
    trim_spines_to_data(ax, xlim=(x_lo, x_hi), ylim=(y_lo, y_hi))
    return True


def plot_panel(
    ax,
    combined: pd.DataFrame,
    title: str,
    structures: list[str],
    print_prefix: str,
) -> bool:
    """Draw one robot's primitive-vs-structure scatter into `ax`, from `combined` (rows indexed
    arbitrarily, one column per structure plus "primitive"). Returns False (and leaves `ax` blank)
    if there's no instance solved under both the primitive baseline and any of `structures`."""
    pairs = {
        name: combined[[BASELINE, name]].dropna()
        for name in structures
        if name in combined.columns
    }
    pairs = {name: pair for name, pair in pairs.items() if not pair.empty}

    for name in structures:
        n = len(pairs.get(name, []))
        print(
            f"{print_prefix}{name}: {n} instances solved under both primitive and {name}"
        )

    groups = [
        (LABELS[name], COLORS[name], pair[BASELINE], pair[name])
        for name, pair in pairs.items()
    ]
    return _draw_scatter_panel(ax, groups, title)


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

    if args.joint:
        fig, ax = plt.subplots(figsize=(6, 6))
        combined = pivot.loc[robots]
        title = (
            "All robots"
            if set(robots) == set(ROBOT_ORDER)
            else " + ".join(ROBOT_LABELS[r] for r in robots)
        )
        plotted_axes = (
            [ax]
            if plot_panel(ax, combined, title, args.structures, "combined/")
            else []
        )
    else:
        ncols = 2 if len(robots) > 1 else 1
        nrows = math.ceil(len(robots) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
        )
        flat_axes = axes.flat

        plotted_axes = []
        for robot, ax in zip(robots, flat_axes):
            combined = pivot.loc[robot]
            if plot_panel(
                ax, combined, ROBOT_LABELS[robot], args.structures, f"{robot}/"
            ):
                plotted_axes.append(ax)

        # Blank any leftover panel when the robot count doesn't fill the grid (e.g. 3 robots in a
        # 2x2 grid).
        for ax in list(flat_axes)[len(robots) :]:
            ax.axis("off")

    fig.text(0.5, 0.065, "Planning time, primitives only (ms)", ha="center")
    if len(args.structures) == 1:
        y_label = f"{LABELS[args.structures[0]]} time (ms)"
    else:
        y_label = "Structure time (ms)"
    fig.text(0.02, 0.55, y_label, va="center", rotation="vertical")

    # One shared legend below the whole figure rather than repeating it per panel, with the
    # equal-time reference line last since it's a guide, not a data series. Collected across all
    # plotted panels (not just the first) and deduplicated, since a robot missing one structure
    # would otherwise drop that series from the legend if it happened to be first.
    seen = {}
    for ax in plotted_axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    if seen:
        labels = sorted(seen, key=lambda label: label == "Equal time")
        handles = [seen[label] for label in labels]
        fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=9,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, 0.0),
        )

    # Bottom margin (0.06) reserves room for the figure-level legend anchored at y=0 above, so the
    # tight/zero-padding crop in `save_figure` doesn't clip it - a negative anchor (placing the
    # legend below the tight_layout rect entirely, as rumple's own version of this script does)
    # relies on that save path's default pad_inches for clearance, which this repo's convention
    # sets to 0.
    fig.tight_layout(rect=(0.03, 0.06, 1, 1))
    save_figure(fig, args.output, crop=True)


if __name__ == "__main__":
    main()
