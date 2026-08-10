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
from matplotlib.ticker import FuncFormatter
from mbm_common import (
    ROBOT_COLORS,
    ROBOT_LABELS,
    YLABEL_PAD,
    finish_single_panel,
    legend_order,
    save_figure,
    style_legend,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "voxel_width_sweep.csv"
OUT = ROOT / "doc" / "voxel_width_sweep.svg"

# Fixed top-to-bottom legend order.
LEGEND_ORDER = [
    ROBOT_LABELS["fetch"],
    ROBOT_LABELS["panda"],
    ROBOT_LABELS["ur5"],
    ROBOT_LABELS["baxter"],
    r"$r_\text{mobile}$",
    r"$r_\text{max}$",
    r"$r_\text{query}$",
]


def main() -> None:
    df = pd.read_csv(RESULTS)
    df = df[df.ns_per_query > 0]
    df["voxel_width_cm"] = df.voxel_width * 100

    fig, ax = plt.subplots(figsize=(5, 4.5))

    for robot, sub in df.groupby("robot"):
        color = ROBOT_COLORS.get(robot, "#000000")
        label = ROBOT_LABELS.get(robot, robot)
        swept = sub[sub.marker == "none"].sort_values("voxel_width_cm")
        mobile_max_row = sub[sub.marker == "mobile_max"]
        query_max_row = sub[sub.marker == "query_max"]
        robot_max_row = sub[sub.marker == "robot_max"]

        if swept.empty:
            print(f"warning: no swept voxel-width data for {robot}, skipping")
            continue

        min_row = sub.loc[sub["ns_per_query"].idxmin()]
        print(
            f"best width for {robot} is {float(min_row.voxel_width)} ({float(min_row.ns_per_query)} ns/q)"
        )

        (line,) = ax.plot(
            swept.voxel_width_cm,
            swept.ns_per_query,
            color=color,
            linewidth=2,
            label=label,
        )
        if not mobile_max_row.empty:
            ax.scatter(
                mobile_max_row.voxel_width_cm,
                mobile_max_row.ns_per_query,
                color=color,
                marker="o",
                s=60,
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
                label=f"{label} (mobile_max)",
            )
        if not robot_max_row.empty:
            ax.scatter(
                robot_max_row.voxel_width_cm,
                robot_max_row.ns_per_query,
                color=color,
                marker="s",
                s=60,
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
                label=f"{label} (robot_max)",
            )
        if not query_max_row.empty:
            ax.scatter(
                query_max_row.voxel_width_cm,
                query_max_row.ns_per_query,
                color=color,
                marker="^",
                s=80,
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
                label=f"{label} (query_max)",
            )

    if not ax.get_legend_handles_labels()[0]:
        print("warning: no voxel-width sweep data for any robot, nothing to plot")
        plt.close(fig)
        return

    ax.set_ylabel("Average query time (ns)", labelpad=YLABEL_PAD)

    # De-duplicate the legend so each robot shows one line-color swatch and each marker shape is
    # explained once, rather than once per robot. Markers use `next(..., None)` since a given
    # marker kind may not have survived for any robot (e.g. all rows for it were missing or
    # non-positive), in which case that legend entry is simply omitted rather than crashing.
    handles, labels = ax.get_legend_handles_labels()
    mobile_max_handle = next(
        (h for h, l in zip(handles, labels) if l.endswith("(mobile_max)")), None
    )
    true_max_handle = next(
        (h for h, l in zip(handles, labels) if l.endswith("(query_max)")), None
    )
    robot_max_handle = next(
        (h for h, l in zip(handles, labels) if l.endswith("(robot_max)")), None
    )
    line_handles = [
        h
        for h, l in zip(handles, labels)
        if not l.endswith(("(mobile_max)", "(query_max)", "(robot_max)"))
    ]
    line_labels = [
        l
        for l in labels
        if not l.endswith(("(mobile_max)", "(query_max)", "(robot_max)"))
    ]

    extra_handles, extra_labels = [], []
    if mobile_max_handle is not None:
        extra_handles.append(mobile_max_handle)
        extra_labels.append(r"$r_\text{mobile}$")
    if robot_max_handle is not None:
        extra_handles.append(robot_max_handle)
        extra_labels.append(r"$r_\text{max}$")
    if true_max_handle is not None:
        extra_handles.append(true_max_handle)
        extra_labels.append(r"$r_\text{query}$")

    finish_single_panel(ax, "Voxel width (cm)", yscale="linear")
    handles, labels = legend_order(
        [*line_handles, *extra_handles], [*line_labels, *extra_labels], LEGEND_ORDER
    )
    style_legend(ax, handles, labels)
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
