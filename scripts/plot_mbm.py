#!/usr/bin/env python3
"""Plot construction/memory/query throughput on MotionBenchMaker workloads.

Reads `data/mbm_bench_results.csv` (produced by `cargo run --release -p mvtable-bench --bin
mbm_bench`) and produces a 5-panel figure (construction time, memory, and average query time for
all/colliding/non-colliding queries vs. point cloud size).

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
from mbm_common import binned_line, density_hexbin, drop_unreliable_query_rows, lighten

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
    "mvtable": "Mvt",
    "mvtable_mutable": "MutableMvt",
    "capt": "CAPT",
    "kiddo": "kiddo",
}
SIMD_COLORS = {name: lighten(color) for name, color in COLORS.items()}

SIMD_LANES = 8
# `kiddo` has no SIMD-batched query API, so it only ever has `lanes == 1` rows.
SIMD_CAPABLE = {"mvtable", "mvtable_mutable", "capt"}

QUERY_PANELS = [
    ("all", "Query Time, All Queries"),
    ("colliding", "Query Time, Colliding Queries"),
    ("non_colliding", "Query Time, Non-Colliding Queries"),
]


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
        "--out",
        type=pathlib.Path,
        default=None,
        help="Output SVG path (default: doc/mbm_throughput.svg if every structure is selected, "
        "otherwise doc/mbm_throughput_<structures>.svg).",
    )
    args = parser.parse_args()

    unknown = set(args.structures) - set(ALL_STRUCTURES)
    if unknown:
        parser.error(
            f"unknown structure(s): {', '.join(sorted(unknown))}; choose from {ALL_STRUCTURES}"
        )
    if not args.structures:
        parser.error("--structures can't be empty")

    if args.out is None:
        if set(args.structures) == set(ALL_STRUCTURES):
            args.out = ROOT / "doc" / "mbm_throughput.svg"
        else:
            args.out = ROOT / "doc" / f"mbm_throughput_{'-'.join(args.structures)}.svg"
    return args


def plot_construction(ax, df: pd.DataFrame, structures: list) -> None:
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
        binned_line(ax, sub.n_points, sub.ms, COLORS[name], annotate=False)
    ax.set_title("Construction Time")
    ax.set_ylabel("Time (Milliseconds)")


def plot_memory(ax, df: pd.DataFrame, structures: list) -> None:
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
        binned_line(ax, sub.n_points, sub.kib, COLORS[name], annotate=False)
    ax.set_title("Memory Consumption")
    ax.set_ylabel("Memory (KiB)")


def plot_query_panel(
    ax,
    df: pd.DataFrame,
    structures: list,
    metric: str,
    title: str,
    extent,
    is_first: bool,
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
    ax.set_title(title)
    ax.set_ylabel("Time (Nanoseconds)" if is_first else "")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(RESULTS)
    df = df[df.structure.isin(args.structures)]
    df = drop_unreliable_query_rows(df)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    plot_construction(axes[0], df, args.structures)
    plot_memory(axes[1], df, args.structures)

    # share one y-extent across all three query-time panels so they stay directly comparable.
    query_metrics = [metric for metric, _ in QUERY_PANELS]
    query_all = df[df.metric.isin(query_metrics)]
    extent = (
        query_all.n_points.min(),
        query_all.n_points.max(),
        query_all.ns_per_op.min(),
        query_all.ns_per_op.max(),
    )
    for i, (ax, (metric, title)) in enumerate(zip(axes[2:], QUERY_PANELS)):
        plot_query_panel(
            ax, df, args.structures, metric, title, extent, is_first=(i == 0)
        )
    axes[5].set_visible(False)  # only 5 panels are used in the 2x3 grid.

    handles, labels = axes[2].get_legend_handles_labels()
    fig.supxlabel("Number of Points in Pointcloud", y=0.04)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("")
        sns.despine(ax=ax)

    n_robots = df.dataset.apply(lambda s: s.split("/")[0]).nunique()
    n_workloads = df.dataset.nunique()
    fig.suptitle(
        "Construction/memory/query throughput on real MotionBenchMaker motion-planning workloads "
        f"({n_robots} robots, {n_workloads} benchmark environments)"
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    args.out.parent.mkdir(exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=150)
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
