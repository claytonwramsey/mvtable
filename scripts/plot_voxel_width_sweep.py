#!/usr/bin/env python3
"""Plot `mvtable::Mvt`'s per-robot voxel-width hyperparameter sweep.

Reads `data/voxel_width_sweep.csv` (written by `mbm_bench`) and writes
`doc/voxel_width_sweep.svg` (+ `.png`).
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from mbm_common import ROBOT_LABELS, save_figure, trim_spines_to_data

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "voxel_width_sweep.csv"
OUT = ROOT / "doc" / "voxel_width_sweep.svg"

# Negative labelpad, pulled in from matplotlib's default (4pt) so the y-axis label sits closer to
# the tick numbers instead of leaving a wide gap of unused margin to its left - see plot_mbm.py's
# YLABEL_PAD, which this matches.
YLABEL_PAD = -8

# Okabe-Ito colorblind-safe palette, one color per robot (distinct from the structure colors used
# in plot_mbm.py, since this plot's series are robots, not structures).
ROBOT_COLORS = {
    "panda": "#0072B2",
    "ur5": "#009E73",
    "fetch": "#D55E00",
    "baxter": "#CC79A7",
}


def main() -> None:
    df = pd.read_csv(RESULTS)
    df = df[df.ns_per_query > 0]
    df["voxel_width_cm"] = df.voxel_width * 100

    fig, ax = plt.subplots(figsize=(5, 4.5))

    for robot, sub in df.groupby("robot"):
        color = ROBOT_COLORS.get(robot, "#000000")
        label = ROBOT_LABELS.get(robot, robot)
        swept = sub[sub.is_r_max == 0].sort_values("voxel_width_cm")
        r_max_row = sub[sub.is_r_max == 1]

        (line,) = ax.plot(
            swept.voxel_width_cm,
            swept.ns_per_query,
            color=color,
            linewidth=2,
            label=label,
        )
        if not r_max_row.empty:
            ax.scatter(
                r_max_row.voxel_width_cm,
                r_max_row.ns_per_query,
                color=color,
                marker="o",
                s=40,
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
                label=f"{label} (r_max)",
            )

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Voxel width (cm)")
    ax.set_ylabel("Average query time (ns)", labelpad=YLABEL_PAD)

    # De-duplicate the legend so each robot shows one line-color swatch and the circle marker is
    # explained once, rather than once per robot.
    handles, labels = ax.get_legend_handles_labels()
    r_max_handle = next(h for h, l in zip(handles, labels) if l.endswith("(r_max)"))
    line_handles = [h for h, l in zip(handles, labels) if not l.endswith("(r_max)")]
    line_labels = [l for l in labels if not l.endswith("(r_max)")]
    ax.legend(
        [*line_handles, r_max_handle],
        [*line_labels, "$r_\\text{max}$"],
        frameon=False,
    )

    sns.despine(ax=ax)
    trim_spines_to_data(ax)
    thin_tick_labels(ax.yaxis)
    fig.tight_layout()
    save_figure(fig, OUT)


def thin_tick_labels(axis) -> None:
    """Blank the text on every other tick of `axis`, keeping the tick marks themselves (and both
    range-frame endpoints, at positions 0 and -1) so the tick density set by `trim_spines_to_data`
    is undisturbed but only every other label is drawn."""
    base_formatter = axis.get_major_formatter()
    n = len(axis.get_majorticklocs())

    def sparse(x, pos=None):
        if pos is not None and pos % 2 == 1 and pos != n - 1:
            return ""
        return base_formatter(x, pos)

    axis.set_major_formatter(FuncFormatter(sparse))


if __name__ == "__main__":
    main()
