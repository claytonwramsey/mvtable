#!/usr/bin/env python3
"""Plot the `MutableMvt`/`Mvt` ratio directly against point cloud size on real MotionBenchMaker
workloads, rather than comparing two absolute-value trend lines (see `plot_mbm.py` for that).

A plot of two absolute-value curves can visually suggest a growing gap (in nanoseconds or bytes)
even while the *relative* cost is shrinking, if the baseline itself is growing too - exactly what
happens for `MutableMvt`'s memory/construction overhead here. Plotting the ratio directly answers
"is this relatively better or worse at scale" without that ambiguity, and makes any crossover
below/above 1.0 immediately visible.

Both axes are log-scaled: point cloud size is heavily right-skewed in this dataset, and the ratio
itself spans almost an order of magnitude (~0.15x to ~4x) across panels.

Reads `data/mbm_bench_results.csv` and writes `doc/mvtable_mutable_ratio_scaling.svg` (+ `.png`).
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mbm_common import binned_line, density_hexbin, drop_unreliable_query_rows

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "mbm_bench_results.csv"
OUT = ROOT / "doc" / "mvtable_mutable_ratio_scaling.svg"

PANELS = [
    ("construction", 1, "Construction Time"),
    ("memory", 1, "Memory Consumption"),
    ("all", 1, "Query Time, All (scalar)"),
    ("all", 8, "Query Time, All (SIMD x8)"),
    ("colliding", 1, "Query Time, Colliding (scalar)"),
    ("colliding", 8, "Query Time, Colliding (SIMD x8)"),
]

COLOR = "#D55E00"


def ratio_for(df: pd.DataFrame, metric: str, lanes: int) -> pd.DataFrame:
    sub = df[(df.metric == metric) & (df.lanes == lanes)]
    mvt = sub[sub.structure == "mvtable"].set_index(["dataset", "filter", "n_points"])["ns_per_op"]
    mut = sub[sub.structure == "mvtable_mutable"].set_index(["dataset", "filter", "n_points"])[
        "ns_per_op"
    ]
    merged = mvt.to_frame("mvt").join(mut.to_frame("mut"), how="inner").reset_index()
    merged = merged[(merged.mvt > 0) & (merged.mut > 0)]
    merged["ratio"] = merged["mut"] / merged["mvt"]
    return merged


def main() -> None:
    df = pd.read_csv(RESULTS)
    df = df[df.structure.isin(["mvtable", "mvtable_mutable"])]
    df = drop_unreliable_query_rows(df)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (metric, lanes, title) in zip(axes, PANELS):
        merged = ratio_for(df, metric, lanes)
        if merged.empty:
            ax.set_visible(False)
            continue

        extent = (merged.n_points.min(), merged.n_points.max(), merged.ratio.min(), merged.ratio.max())
        density_hexbin(ax, merged.n_points, merged.ratio, COLOR, extent)
        binned_line(ax, merged.n_points, merged.ratio, COLOR)
        ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("MutableMvt / Mvt ratio" if ax is axes[0] or ax is axes[3] else "")
        sns.despine(ax=ax)

    fig.supxlabel("Number of Points in Pointcloud", y=0.02)
    n_robots = df.dataset.apply(lambda s: s.split("/")[0]).nunique()
    n_workloads = df.dataset.nunique()
    fig.suptitle(
        "MutableMvt / Mvt ratio vs. point cloud size, real MotionBenchMaker workloads "
        f"({n_robots} robots, {n_workloads} benchmark environments) — dotted line is parity"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    print(f"wrote {OUT}")
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
