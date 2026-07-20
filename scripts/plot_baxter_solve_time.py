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
from mbm_common import (
    STRUCTURE_COLORS,
    YLABEL_PAD,
    lighten,
    save_figure,
    trim_spines_to_data,
    wrap_tick_label,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "mbm_plan_results.csv"

ALL_STRUCTURES = [
    "mvtable",
    "mvtable_simd",
    "mvtable_mutable",
    "mvtable_mutable_simd",
    "mvtable_cpp",
    "mvt_cpp_simd",
    "capt",
    "capt_simd",
    "kiddo",
]
# `mbm_plan_results.csv` names the C++ port "mvtable_cpp" (`mbm_bench_results.csv`, read by
# plot_mbm.py, calls the same structure "mvt_cpp") - same shared `STRUCTURE_COLORS` color, just
# re-keyed to match this file's own naming.
COLORS = {
    "mvtable": STRUCTURE_COLORS["mvtable"],
    "mvtable_simd": lighten(STRUCTURE_COLORS["mvtable"]),
    "mvtable_mutable": STRUCTURE_COLORS["mvtable_mutable"],
    "mvtable_mutable_simd": lighten(STRUCTURE_COLORS["mvtable_mutable"]),
    "mvtable_cpp": STRUCTURE_COLORS["mvt_cpp"],
    "mvt_cpp_simd": lighten(STRUCTURE_COLORS["mvt_cpp"]),
    "capt": STRUCTURE_COLORS["capt"],
    "capt_simd": lighten(STRUCTURE_COLORS["capt"]),
    "kiddo": STRUCTURE_COLORS["kiddo"],
}
LABELS = {
    "mvtable": "MVT",
    "mvtable_simd": "MVT (SIMD)",
    "mvtable_mutable": "Mutable MVT",
    "mvtable_mutable_simd": "Mutable MVT (SIMD)",
    "mvtable_cpp": "MVT (C++)",
    "mvt_cpp_simd": "MVT (C++, SIMD)",
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

    fig, ax = plt.subplots(figsize=(0.9 * len(args.structures), 4.5))
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
    ax.set_ylabel("Solve Time (Milliseconds)", labelpad=YLABEL_PAD)
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
