#!/usr/bin/env python3
"""Plot end-to-end motion-planning-time distributions across collision-checking backends.

Reads `data/mbm_plan_results.csv` (produced by `cargo run --release -p mbm-plan-bench`).

Produces a two-row figure: a violin plot of solve-time distributions faceted by
robot with backend as hue, and a solve-rate bar chart underneath it for context.

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
from mbm_common import lighten

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
    "mvtable": "Mvt",
    "mvtable_simd": "Mvt SIMD",
    "mvtable_mutable": "MutableMvt",
    "mvtable_mutable_simd": "MutableMvt SIMD",
    "capt": "CAPT",
    "capt_simd": "CAPT SIMD",
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
        cut=0,
        linewidth=0.75,
        log_scale=True,
    )
    ax.set_ylabel("Solve Time (Seconds)")
    ax.set_xlabel("")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [], [], frameon=False
    )  # the figure-level legend below carries this instead.
    return handles, [LABELS[name] for name in labels]


def plot_solve_rate(ax, df: pd.DataFrame, structures: list) -> None:
    robots = [r for r in ROBOT_ORDER if r in df.robot.unique()]
    rate = (
        df.groupby(["robot", "structure"], observed=True)
        .solved.mean()
        .mul(100)
        .rename("solve_rate")
        .reset_index()
    )
    sns.barplot(
        data=rate,
        x="robot",
        y="solve_rate",
        hue="structure",
        order=robots,
        hue_order=structures,
        palette=COLORS,
        ax=ax,
    )
    ax.set_ylabel("Solve Rate (%)")
    ax.set_xlabel("Robot")
    ax.set_ylim(0, 100)
    ax.legend([], [], frameon=False)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(RESULTS)
    df = df[df.structure.isin(args.structures)]

    time_col = "time_secs"
    if args.include_construction:
        df = df.copy()
        df["total_secs"] = df.time_secs + df.construction_secs
        time_col = "total_secs"

    fig, (ax_time, ax_rate) = plt.subplots(
        2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True
    )

    handles, labels = plot_solve_time_violins(ax_time, df, args.structures, time_col)
    plot_solve_rate(ax_rate, df, args.structures)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    for ax in (ax_time, ax_rate):
        sns.despine(ax=ax)

    title_metric = (
        "total (construction + solve) time"
        if args.include_construction
        else "solve time"
    )
    fig.suptitle(f"Solve Time by Collision Checking Backend\n{title_metric}")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    args.out.parent.mkdir(exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=150)
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
