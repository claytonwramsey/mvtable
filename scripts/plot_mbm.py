#!/usr/bin/env python3
"""Plot construction/memory/query throughput on MotionBenchMaker workloads.

Reads `data/mbm_bench_results.csv` (produced by `cargo run --release -p mvtable-bench --bin
mbm_bench`) and produces three figures: construction time, memory consumption, and query time.
With `--titles` (used for the README) the query figure is a 3-panel (all/colliding/non-colliding
queries) breakdown sharing a y-axis; without it (the default, for the blog) it's just the
all-queries panel, which reads better at blog width.

Use `--structures` to pick which ones appear:

    python3 scripts/plot_mbm.py                              # all four structures
    python3 scripts/plot_mbm.py --structures mvtable,capt,kiddo    # the general comparison
    python3 scripts/plot_mbm.py --structures mvtable,mvtable_mutable   # the mutable-vs-immutable one

Both axes are log-scaled on every panel: point cloud size spans 1 to ~15,000 points in this
dataset (heavily right-skewed - most real, filtered clouds are small), and every timing/memory
metric spans a comparable range, so a linear scale either crushes the small end into a sliver or
cuts off the large end.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mbm_common import (
    binned_line,
    density_hexbin,
    drop_unreliable_query_rows,
    lighten,
    save_figure,
    trim_spines_to_data,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "mbm_bench_results.csv"

ALL_STRUCTURES = ["mvtable", "mvtable_mutable", "capt", "kiddo"]
COLORS = {
    "mvtable": "#0072B2",
    "mvtable_mutable": "#D55E00",
    "capt": "#009E73",
    "kiddo": "#E69F00",
}
LABELS = {
    "mvtable": "MVT",
    "mvtable_mutable": "Mutable MVT",
    "capt": "CAPT",
    "kiddo": "kiddo",
}
SIMD_COLORS = {name: lighten(color) for name, color in COLORS.items()}

SIMD_LANES = 8
# `kiddo` has no SIMD-batched query API, so it only ever has `lanes == 1` rows.
SIMD_CAPABLE = {"mvtable", "mvtable_mutable", "capt"}

# left-to-right order of the query panels in the 3-panel query figure.
QUERY_METRICS = ["all", "colliding", "non_colliding"]
QUERY_TITLES = {
    "all": "Query Time, All Queries",
    "colliding": "Query Time, Colliding Queries",
    "non_colliding": "Query Time, Non-Colliding Queries",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--structures",
        type=lambda s: s.split(","),
        default=ALL_STRUCTURES,
        help=f"Comma-separated subset of {{{','.join(ALL_STRUCTURES)}}} to plot (default: all). "
        "Fewer structures make a busy figure easier to read.",
    )
    parser.add_argument(
        "--out-prefix",
        type=pathlib.Path,
        default=None,
        help="Base path for the three output SVGs, suffixed with '_construction.svg', "
        "'_memory.svg', and '_query.svg' (+ matching .png files). Default: "
        "doc/mbm_throughput if every structure is selected, otherwise "
        "doc/mbm_throughput_<structures>.",
    )
    parser.add_argument(
        "--titles",
        action="store_true",
        help="Add per-panel chart titles. Off by default so the SVGs drop cleanly into a page "
        "that supplies its own captions.",
    )
    args = parser.parse_args()

    unknown = set(args.structures) - set(ALL_STRUCTURES)
    if unknown:
        parser.error(
            f"unknown structure(s): {', '.join(sorted(unknown))}; choose from {ALL_STRUCTURES}"
        )
    if not args.structures:
        parser.error("--structures can't be empty")

    if args.out_prefix is None:
        if set(args.structures) == set(ALL_STRUCTURES):
            args.out_prefix = ROOT / "doc" / "mbm_throughput"
        else:
            args.out_prefix = (
                ROOT / "doc" / f"mbm_throughput_{'-'.join(args.structures)}"
            )
    return args


def plot_construction(
    ax, df: pd.DataFrame, structures: list, title: str | None
) -> None:
    construction = df[df.metric == "construction"].copy()
    construction["ms"] = construction.ns_per_op / 1e6
    extent = (
        construction.n_points.min(),
        construction.n_points.max(),
        construction.ms.min(),
        construction.ms.max(),
    )
    for name in structures:
        sub = construction[construction.structure == name]
        if sub.empty:
            continue
        density_hexbin(ax, sub.n_points, sub.ms, COLORS[name], extent)
        binned_line(
            ax, sub.n_points, sub.ms, COLORS[name], annotate=False, label=LABELS[name]
        )
    ax.set_ylabel("Construction time (milliseconds)")
    if title:
        ax.set_title(title)


def plot_memory(ax, df: pd.DataFrame, structures: list, title: str | None) -> None:
    memory = df[df.metric == "memory"].copy()
    memory["kib"] = (
        memory.ns_per_op / 1024
    )  # `ns_per_op` holds bytes for the `memory` metric.
    extent = (
        memory.n_points.min(),
        memory.n_points.max(),
        memory.kib.min(),
        memory.kib.max(),
    )
    for name in structures:
        sub = memory[memory.structure == name]
        if sub.empty:
            continue
        density_hexbin(ax, sub.n_points, sub.kib, COLORS[name], extent)
        binned_line(
            ax, sub.n_points, sub.kib, COLORS[name], annotate=False, label=LABELS[name]
        )
    ax.set_ylabel("Memory (KiB)")
    if title:
        ax.set_title(title)


def plot_query_panel(
    ax,
    df: pd.DataFrame,
    structures: list,
    metric: str,
    extent,
    is_first: bool,
    title: str | None,
) -> None:
    query = df[df.metric == metric]
    for name in structures:
        for lanes, color, linestyle in [
            (1, COLORS[name], "-"),
            *([(SIMD_LANES, SIMD_COLORS[name], "--")] if name in SIMD_CAPABLE else []),
        ]:
            sub = query[(query.structure == name) & (query.lanes == lanes)]
            if sub.empty:
                continue
            # skip the density cloud for a structure's scalar series when it also has a SIMD one,
            # so the two overlapping point clouds don't just paint over each other.
            if lanes != 1 or name not in SIMD_CAPABLE:
                density_hexbin(ax, sub.n_points, sub.ns_per_op, color, extent)
            label = (
                LABELS[name] if lanes == 1 else f"{LABELS[name]} (SIMD x{SIMD_LANES})"
            )
            binned_line(
                ax,
                sub.n_points,
                sub.ns_per_op,
                color,
                linestyle=linestyle,
                annotate=False,
                label=label,
            )
    if is_first:
        ax.set_ylabel("Time (Nanoseconds)")
    if title:
        ax.set_title(title)


def finish_single_panel(ax, xlabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    sns.despine(ax=ax)
    trim_spines_to_data(ax)


def legend_order(handles, labels, pin_first: str | None = None):
    """Reorder legend entries to match the plotted lines' relative vertical order at the left
    edge of the plot (where a left-to-right reader looks first), so e.g. the highest series
    (CAPT, for construction time) reads first/top in the legend rather than in a fixed
    structure order.

    `pin_first`, if given, forces that label to the top regardless of its left-edge position -
    for the memory panel, where CAPT starts close to the pack but leads for nearly the whole
    plot, so top-of-legend is the more honest read than "whatever's highest at x=0"."""
    order = sorted(range(len(handles)), key=lambda i: handles[i].get_ydata()[0], reverse=True)
    if pin_first is not None:
        order = sorted(order, key=lambda i: labels[i] != pin_first)
    return [handles[i] for i in order], [labels[i] for i in order]


def save_single_panel(
    fig, ax, xlabel: str, out: pathlib.Path, crop: bool, pin_legend_first: str | None = None
) -> None:
    finish_single_panel(ax, xlabel)
    handles, labels = legend_order(*ax.get_legend_handles_labels(), pin_first=pin_legend_first)
    ax.legend(handles, labels, frameon=False)
    fig.tight_layout()
    save_figure(fig, out, crop=crop)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(RESULTS)
    df = df[df.structure.isin(args.structures)]
    df = drop_unreliable_query_rows(df)

    xlabel = "Number of points in point cloud"
    crop = not args.titles

    # construction time
    fig, ax = plt.subplots(figsize=(5, 4.5))
    plot_construction(
        ax, df, args.structures, title="Construction Time" if args.titles else None
    )
    save_single_panel(
        fig,
        ax,
        xlabel,
        args.out_prefix.parent / f"{args.out_prefix.name}_construction.svg",
        crop,
    )

    # memory consumption
    fig, ax = plt.subplots(figsize=(5, 4.5))
    plot_memory(
        ax, df, args.structures, title="Memory Consumption" if args.titles else None
    )
    save_single_panel(
        fig,
        ax,
        xlabel,
        args.out_prefix.parent / f"{args.out_prefix.name}_memory.svg",
        crop,
        pin_legend_first=LABELS["capt"],
    )

    query_out = args.out_prefix.parent / f"{args.out_prefix.name}_query.svg"
    if not args.titles:
        # the blog embeds figures at a width where the 3-panel breakdown below is hard to read;
        # just the all-queries panel carries the same headline result more legibly.
        all_queries = df[df.metric == "all"]
        extent = (
            all_queries.n_points.min(),
            all_queries.n_points.max(),
            all_queries.ns_per_op.min(),
            all_queries.ns_per_op.max(),
        )
        fig, ax = plt.subplots(figsize=(5, 4.5))
        plot_query_panel(
            ax, df, args.structures, "all", extent, is_first=True, title=None
        )
        save_single_panel(fig, ax, xlabel, query_out, crop)
        return

    # query time, 3 panels (all / colliding / non-colliding) sharing one y-axis, for the README
    # where there's room (and surrounding prose) to explain the breakdown.
    query_all = df[df.metric.isin(QUERY_METRICS)]
    extent = (
        query_all.n_points.min(),
        query_all.n_points.max(),
        query_all.ns_per_op.min(),
        query_all.ns_per_op.max(),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for i, (ax, metric) in enumerate(zip(axes, QUERY_METRICS)):
        plot_query_panel(
            ax,
            df,
            args.structures,
            metric,
            extent,
            is_first=(i == 0),
            title=QUERY_TITLES[metric],
        )
        finish_single_panel(ax, "")
    handles, labels = legend_order(*axes[0].get_legend_handles_labels())
    fig.supxlabel(xlabel, y=0.055)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    save_figure(fig, query_out)


if __name__ == "__main__":
    main()
