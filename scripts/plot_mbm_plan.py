#!/usr/bin/env python3
"""Plot end-to-end motion-planning-time distributions across collision-checking backends.

Reads `data/mbm_plan_results.csv` (produced by `cargo run --release -p mbm-plan-bench`).

Produces a violin plot of solve-time distributions faceted by robot with backend as hue. See
`plot_baxter_solve_time.py` for the blog's single-robot variant of this figure.

    python3 scripts/plot_mbm_plan.py
    python3 scripts/plot_mbm_plan.py --structures mvtable,capt
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mbm_common import ROBOT_LABELS, lighten, save_figure, trim_spines_to_data

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "mbm_plan_results.csv"

ALL_STRUCTURES = [
    "mvtable",
    "mvtable_simd",
    "mvtable_mutable",
    "mvtable_mutable_simd",
    "capt",
    "capt_simd",
    "kiddo",
]
COLORS = {
    "mvtable": "#0072B2",
    "mvtable_simd": lighten("#0072B2"),
    "mvtable_mutable": "#D55E00",
    "mvtable_mutable_simd": lighten("#D55E00"),
    "capt": "#009E73",
    "capt_simd": lighten("#009E73"),
    "kiddo": "#E69F00",
}
LABELS = {
    "mvtable": "MVT",
    "mvtable_simd": "MVT (SIMD x8)",
    "mvtable_mutable": "Mutable MVT",
    "mvtable_mutable_simd": "Mutable MVT (SIMD x8)",
    "capt": "CAPT",
    "capt_simd": "CAPT (SIMD x8)",
    "kiddo": "kiddo",
}
ROBOT_ORDER = ["panda", "ur5", "fetch", "baxter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--structures",
        type=lambda s: s.split(","),
        default=ALL_STRUCTURES,
        help=f"Comma-separated subset of {{{','.join(ALL_STRUCTURES)}}} to plot (default: all).",
    )
    parser.add_argument(
        "--include-construction",
        action="store_true",
        help="Add each backend's structure-construction time to its solve time, showing total "
        "wall-clock cost rather than pure planning-query cost.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="Output SVG path (default: doc/mbm_plan_times.svg, or "
        "doc/mbm_plan_times_<structures>.svg if a subset of structures is selected).",
    )
    parser.add_argument(
        "--titles",
        action="store_true",
        help="Add a chart title. Off by default so the SVG drops cleanly into a page that "
        "supplies its own captions.",
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
            args.out = ROOT / "doc" / "mbm_plan_times.svg"
        else:
            args.out = ROOT / "doc" / f"mbm_plan_times_{'-'.join(args.structures)}.svg"
    return args


def plot_solve_time_violins(
    ax, df: pd.DataFrame, structures: list, time_col: str
) -> None:
    solved = df[df.solved]
    robots = [r for r in ROBOT_ORDER if r in solved.robot.unique()]
    sns.violinplot(
        data=solved,
        x="robot",
        y=time_col,
        hue="structure",
        order=robots,
        hue_order=structures,
        palette=COLORS,
        ax=ax,
        density_norm="width",
        cut=0.0,
        linewidth=0.75,
        log_scale=True,
        width=0.9,
    )
    ax.set_ylabel("Solve Time (Milliseconds)")
    ax.set_xlabel("")
    ax.set_xticks(range(len(robots)), [ROBOT_LABELS[r] for r in robots])
    ax.tick_params(axis="x", bottom=False, pad=2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [], [], frameon=False
    )  # the figure-level legend below carries this instead.
    return handles, [LABELS[name] for name in labels]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(RESULTS)
    df = df[df.structure.isin(args.structures)]
    df = df.copy()

    time_col = "time_ms"
    df["time_ms"] = df.time_secs * 1000
    if args.include_construction:
        df["total_ms"] = (df.time_secs + df.construction_secs) * 1000
        time_col = "total_ms"

    fig, ax_time = plt.subplots(figsize=(12, 6))

    handles, labels = plot_solve_time_violins(ax_time, df, args.structures, time_col)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    sns.despine(ax=ax_time, bottom=True)
    trim_spines_to_data(ax_time)

    if args.titles:
        fig.suptitle("Solve Time by Collision Checking Backend")
        fig.tight_layout(rect=(0, 0.045, 1, 0.9))
    else:
        fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_figure(fig, args.out, crop=not args.titles)


if __name__ == "__main__":
    main()
