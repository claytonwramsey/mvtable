#!/usr/bin/env python3
"""Plot the Baxter solve-time distribution by collision-checking backend, for the blog.

Reads `data/mbm_plan_results.csv` (produced by `cargo run --release -p mbm-plan-bench`) and plots
just Baxter, which shows the most pronounced gap between backends. See `plot_mbm_plan.py` for the
README's full 4-robot breakdown.

    python3 scripts/plot_baxter_solve_time.py
    python3 scripts/plot_baxter_solve_time.py --structures mvtable,capt
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mbm_common import lighten, save_figure, trim_spines_to_data, wrap_tick_label

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
    "mvtable_simd": "MVT (SIMD)",
    "mvtable_mutable": "Mutable MVT",
    "mvtable_mutable_simd": "Mutable MVT (SIMD)",
    "capt": "CAPT",
    "capt_simd": "CAPT (SIMD)",
    "kiddo": "kiddo",
}


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
        "--out",
        type=pathlib.Path,
        default=ROOT / "doc" / "baxter_solve_time.svg",
        help="Output SVG path (default: doc/baxter_solve_time.svg).",
    )
    args = parser.parse_args()

    unknown = set(args.structures) - set(ALL_STRUCTURES)
    if unknown:
        parser.error(
            f"unknown structure(s): {', '.join(sorted(unknown))}; choose from {ALL_STRUCTURES}"
        )
    if not args.structures:
        parser.error("--structures can't be empty")
    return args


def main() -> None:
    args = parse_args()
    df = pd.read_csv(RESULTS)
    df = df[df.structure.isin(args.structures)]
    df = df[(df.robot == "baxter") & df.solved]
    df = df.copy()
    df["time_ms"] = df.time_secs * 1000

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.violinplot(
        data=df,
        x="structure",
        y="time_ms",
        hue="structure",
        order=args.structures,
        palette=COLORS,
        ax=ax,
        density_norm="width",
        cut=0,
        linewidth=0.75,
        log_scale=True,
        width=0.9,
        legend=False,
    )
    ax.set_ylabel("Solve Time (Milliseconds)")
    ax.set_xlabel("")
    ax.set_xticks(
        range(len(args.structures)),
        [wrap_tick_label(LABELS[name]) for name in args.structures],
    )
    ax.tick_params(axis="x", bottom=False, pad=2)
    sns.despine(ax=ax, bottom=True)
    trim_spines_to_data(ax)
    fig.tight_layout()
    save_figure(fig, args.out, crop=True)


if __name__ == "__main__":
    main()
