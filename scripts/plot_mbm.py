#!/usr/bin/env python3
"""Plot construction/memory/query throughput on MotionBenchMaker workloads.

Reads `data/mbm_bench_results.csv` (produced by `cargo run --release -p mvtable-bench --bin
mbm_bench`) and produces three figures: construction time, memory consumption, and query time (all
queries - colliding and non-colliding together). `--titles` adds a per-panel chart title to each
figure; without it (the default) the panels are left untitled to drop cleanly into a page that
supplies its own captions.

Use `--structures` to pick which ones appear:

    python3 scripts/plot_mbm.py                              # all four structures
    python3 scripts/plot_mbm.py --structures mvtable,capt,kiddo    # the general comparison
    python3 scripts/plot_mbm.py --structures mvtable,mvtable_mutable   # the mutable-vs-immutable one

The x-axis (point cloud size) is linear; the y-axis (time/memory) is log-scaled, since every
timing/memory metric spans a comparable multi-order-of-magnitude range that a linear scale would
crush into a sliver at the small end.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mbm_common import (
    STRUCTURE_COLORS,
    YLABEL_PAD,
    binned_line,
    drop_unreliable_query_rows,
    finish_single_panel,
    legend_order,
    lighten,
    save_figure,
    style_legend,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "mbm_bench_results.csv"

ALL_STRUCTURES = ["mvtable", "mvtable_mutable", "capt", "kiddo", "mvt_cpp"]
COLORS = STRUCTURE_COLORS
LABELS = {
    "mvtable": "MVT",
    "mvtable_mutable": "Mutable MVT",
    "capt": "CAPT",
    "kiddo": "kiddo",
    "mvt_cpp": "MVT (C++)",
}
SIMD_COLORS = {name: lighten(color) for name, color in COLORS.items()}

SIMD_LANES = 8
# `kiddo` has no SIMD-batched query API, so it only ever has `lanes == 1` rows.
SIMD_CAPABLE = {"mvtable", "mvtable_mutable", "capt", "mvt_cpp"}
# Every structure's SIMD line shares this one generic legend entry (dashed, neutral gray) instead
# of a per-structure "<name> (SIMD)" one - dashed already reads as "SIMD" once it's called out
# once, so repeating it per structure only doubled the query panel's legend for no extra
# information.
SIMD_LEGEND_LABEL = "SIMD"
# The site's own neutral black (`--bh-black` in main.css), not a structure color, since this
# entry marks a linestyle (dashed = SIMD) rather than any one series.
SIMD_LEGEND_COLOR = "#1A1A1A"

# Fixed top-to-bottom legend order for the query panel - kiddo first since it's the only
# non-SIMD-capable baseline, then CAPT, then the three MVT variants, then the generic SIMD entry.
QUERY_LEGEND_ORDER = [
    LABELS["kiddo"],
    LABELS["capt"],
    LABELS["mvtable"],
    LABELS["mvtable_mutable"],
    LABELS["mvt_cpp"],
    SIMD_LEGEND_LABEL,
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
    for name in structures:
        sub = construction[construction.structure == name]
        if sub.empty:
            continue
        binned_line(
            ax, sub.n_points, sub.ms, COLORS[name], annotate=False, label=LABELS[name]
        )
    ax.set_ylabel("Construction time (ms)", labelpad=YLABEL_PAD)
    if title:
        ax.set_title(title)


def plot_memory(ax, df: pd.DataFrame, structures: list, title: str | None) -> None:
    memory = df[df.metric == "memory"].copy()
    memory["kib"] = (
        memory.ns_per_op / 1024
    )  # `ns_per_op` holds bytes for the `memory` metric.
    for name in structures:
        sub = memory[memory.structure == name]
        if sub.empty:
            continue
        binned_line(
            ax, sub.n_points, sub.kib, COLORS[name], annotate=False, label=LABELS[name]
        )
    ax.set_ylabel("Memory (KiB)", labelpad=YLABEL_PAD)
    if title:
        ax.set_title(title)


def plot_query_panel(
    ax,
    df: pd.DataFrame,
    structures: list,
    metric: str,
    is_first: bool,
    title: str | None,
) -> None:
    query = df[df.metric == metric]
    any_simd = False
    for name in structures:
        for lanes, color, linestyle in [
            (1, COLORS[name], "-"),
            *([(SIMD_LANES, SIMD_COLORS[name], "--")] if name in SIMD_CAPABLE else []),
        ]:
            sub = query[(query.structure == name) & (query.lanes == lanes)]
            if sub.empty:
                continue
            is_simd = lanes != 1
            any_simd = any_simd or is_simd
            binned_line(
                ax,
                sub.n_points,
                sub.ns_per_op,
                color,
                linestyle=linestyle,
                annotate=False,
                label=None if is_simd else LABELS[name],
            )
    if any_simd:
        ax.plot(
            [], [], color=SIMD_LEGEND_COLOR, linestyle="--", label=SIMD_LEGEND_LABEL
        )
    if is_first:
        ax.set_ylabel("Query time (ns)", labelpad=YLABEL_PAD)
    if title:
        ax.set_title(title)


def save_single_panel(
    fig,
    ax,
    xlabel: str,
    out: pathlib.Path,
    crop: bool,
    pin_legend_first: str | None = None,
    legend_fixed_order: list[str] | None = None,
) -> None:
    finish_single_panel(ax, xlabel)
    handles, labels = legend_order(
        *ax.get_legend_handles_labels(),
        pin_first=pin_legend_first,
        fixed_order=legend_fixed_order,
    )
    style_legend(ax, handles, labels)
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

    # query time (all queries - colliding and non-colliding together)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    plot_query_panel(
        ax,
        df,
        args.structures,
        "all",
        is_first=True,
        title="Query Time" if args.titles else None,
    )
    save_single_panel(
        fig,
        ax,
        xlabel,
        args.out_prefix.parent / f"{args.out_prefix.name}_query.svg",
        crop,
        legend_fixed_order=QUERY_LEGEND_ORDER,
    )


if __name__ == "__main__":
    main()
