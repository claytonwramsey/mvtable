#!/usr/bin/env python3
"""Plot collision-checking throughput on real MotionBenchMaker workloads.

Reads `data/mbm_bench_results.csv` (produced by `cargo run --release -p mvtable-bench --bin
mbm_bench`) and produces a 4-panel figure (construction time, and average query time for all/
colliding/non-colliding queries vs. point cloud size).
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import MaxNLocator

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "mbm_bench_results.csv"
OUT = ROOT / "doc" / "mbm_throughput.svg"

STRUCTURES = ["mvtable", "capt", "kiddo"]
COLORS = {"mvtable": "#0072B2", "capt": "#009E73", "kiddo": "#E69F00"}
LABELS = {"mvtable": "MVT", "capt": "CAPT", "kiddo": "kiddo"}


def lighten(color: str, amount: float = 0.55) -> tuple:
    """Blend `color` toward white by `amount`, for a SIMD series' line color that's visibly
    related to (but distinguishable from) its structure's scalar series' color."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


SIMD_COLORS = {"mvtable": lighten(COLORS["mvtable"]), "capt": lighten(COLORS["capt"])}

# Whether each structure's query-time trendline is fit against log(n_points) or n_points
# directly. `kiddo`'s balanced k-d tree matches its O(log n) theoretical query complexity well
# (checked against this benchmark's own data: R^2=0.82 for a log fit vs. 0.58 for a linear fit).
# `capt`'s theoretical complexity is also O(log n), but that model fits this benchmark's actual
# data poorly (R^2=0.31-0.33 across scalar and SIMD), while a linear fit tracks it much better
# (R^2=0.73-0.75) - not because `capt`'s true complexity is O(n), but because query cost here is
# dominated by other factors (affordance buffer size, per-workload r_range) that happen to
# correlate closely with point count across this benchmark's real, heterogeneous workloads.
# `mvtable`'s voxel-table query complexity is O(n) in point cloud size, matching a linear fit both
# theoretically and empirically.
QUERY_FIT_LOGX = {"mvtable": False, "capt": False, "kiddo": True}

SIMD_LANES = 8

# One entry per line drawn in each query-time panel: (structure, lanes, label, color, linestyle).
# `mvtable` and `capt` each get a scalar (lanes=1) and a SIMD-batched (lanes=SIMD_LANES) series;
# `kiddo` has no SIMD-batched query API, so it only ever has lanes=1 rows.
QUERY_SERIES = [
    ("mvtable", 1, "MVT", COLORS["mvtable"], "-"),
    ("mvtable", SIMD_LANES, f"MVT (SIMD x{SIMD_LANES})", SIMD_COLORS["mvtable"], "--"),
    ("capt", 1, "CAPT", COLORS["capt"], "-"),
    ("capt", SIMD_LANES, f"CAPT (SIMD x{SIMD_LANES})", SIMD_COLORS["capt"], "--"),
    ("kiddo", 1, "kiddo", COLORS["kiddo"], "-"),
]

QUERY_PANELS = [
    ("all", "Average Query Time for All Queries"),
    ("colliding", "Average Query Time for Colliding Queries"),
    ("non_colliding", "Average Query Time for Non-Colliding Queries"),
]


def density_cmap(color) -> LinearSegmentedColormap:
    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list(
        "density", [(r, g, b, 0.0), (r, g, b, 1.0)]
    )


def density_hexbin(
    ax, sub: pd.DataFrame, x: str, y: str, color, yscale: str, extent
) -> None:
    ax.hexbin(
        sub[x],
        sub[y],
        gridsize=22,
        cmap=density_cmap(color),
        mincnt=1,
        linewidths=0.0,
        yscale=yscale,
        bins="log",
        extent=extent,
    )


def main() -> None:
    df = pd.read_csv(RESULTS)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    ax = axes[0]
    construction = df[df.metric == "construction"].copy()
    construction["ms"] = construction.ns_per_op / 1e6
    construction_extent = (
        construction.n_points.min(),
        construction.n_points.max(),
        construction.ms.min(),
        construction.ms.max(),
    )
    for name in STRUCTURES:
        sub = construction[construction.structure == name]
        density_hexbin(
            ax, sub, "n_points", "ms", COLORS[name], "linear", construction_extent
        )
        sns.regplot(
            data=sub,
            x="n_points",
            y="ms",
            order=2,
            ci=99,
            ax=ax,
            color=COLORS[name],
            label=LABELS[name],
            scatter=False,
            line_kws={"linewidth": 2},
        )
    ax.set_title("Construction Time")
    ax.set_xlabel("")
    ax.set_ylabel("Time (Milliseconds)")

    # Share one x/y extent across all three query-time panels (computed from their combined
    # data), so the panels are directly visually comparable rather than each auto-scaling to its
    # own subset's range, and hexbin draws consistently-shaped (not squashed) hexagons on every
    # panel regardless of how widely that panel's own data happens to be spread.
    query_metrics = [metric for metric, _ in QUERY_PANELS]
    query_all = df[df.metric.isin(query_metrics)]
    x_lo, x_hi = query_all.n_points.min(), query_all.n_points.max()
    y_lo, y_hi = query_all.ns_per_op.min(), query_all.ns_per_op.max()
    query_extent = (x_lo, x_hi, y_lo, y_hi)

    for i, (ax, (metric, title)) in enumerate(zip(axes[1:], QUERY_PANELS)):
        query = df[df.metric == metric]
        for name, lanes, label, color, linestyle in QUERY_SERIES:
            sub = query[(query.structure == name) & (query.lanes == lanes)]
            if sub.empty:
                continue
            # Skip the density heatmap for a structure's scalar series when it also has a SIMD
            # series (mvtable, capt).
            if lanes != 1 or name == "kiddo":
                density_hexbin(
                    ax, sub, "n_points", "ns_per_op", color, "linear", query_extent
                )
            # `logx=True` fits ns_per_op ~ a*log(n_points) + b (this structure's theoretical
            # O(log n) query complexity); `logx=False` fits a straight line against n_points
            # directly (mvtable's O(n) query complexity). SIMD-batched series use the same fit
            # as their structure's scalar series - batching changes the constant factor, not the
            # asymptotic complexity.
            sns.regplot(
                data=sub,
                x="n_points",
                y="ns_per_op",
                logx=QUERY_FIT_LOGX[name],
                ci=99,
                ax=ax,
                color=color,
                label=label,
                scatter=False,
                line_kws={"linewidth": 2, "linestyle": linestyle},
            )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("Time (Nanoseconds)" if i == 0 else "")

    # One legend and one x-axis label shared across all four panels. The second panel has every series (including the
    # SIMD ones construction has no separate line for), so its handles cover the whole figure.
    handles, labels = axes[1].get_legend_handles_labels()
    fig.supxlabel("Number of Points in Pointcloud", y=0.135)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    for ax in axes:
        sns.despine(ax=ax)
        # The x-axis otherwise ends up densely tick-labeled given how wide n_points ranges.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    n_robots = df.dataset.apply(lambda s: s.split("/")[0]).nunique()
    n_workloads = df.dataset.nunique()
    fig.suptitle(
        "Construction and query throughput on real MotionBenchMaker motion-planning workloads "
        f"({n_robots} robots, {n_workloads} benchmark environments)"
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
