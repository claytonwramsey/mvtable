"""Shared helpers for the `mbm_bench` result-plotting scripts.

`mbm_bench.rs` computes `ns_per_op` for a `(structure, dataset, filter, n_points, metric, lanes)`
row as `elapsed_ns / trace.len()`, i.e. an average over however many queries happened to land in
that trace. For the `colliding` trace specifically, that count can be tiny: a heavily-filtered or
naturally collision-sparse scene may have only a handful of colliding queries in its 10,000-query
sample (roughly a third of `colliding` rows have fewer than 30). Averaging over a handful of
queries lets a single noisy one dominate the whole row.
That single row is enough to distort a shared y-axis extent computed from raw
min/max across a whole figure. Filtering out rows whose
`n_queries` is too small to average out that kind of noise avoids treating it as a real result.
"""

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, to_rgb

QUERY_METRICS = ["all", "colliding", "non_colliding"]

# filter threshold for rejecting rows with too little data
MIN_QUERIES_FOR_RELIABLE_AVERAGE = 50


def drop_unreliable_query_rows(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove untrustworthy rows with too few queries from a dataframe."""
    is_query_row = df.metric.isin(QUERY_METRICS)
    unreliable = is_query_row & (df.n_queries < MIN_QUERIES_FOR_RELIABLE_AVERAGE)
    if verbose and unreliable.any():
        print(
            f"dropping {unreliable.sum()} / {is_query_row.sum()} query rows with fewer than "
            f"{MIN_QUERIES_FOR_RELIABLE_AVERAGE} queries (unreliable average, see mbm_common.py)"
        )
    return df[~unreliable].copy()


def geomean(x: pd.Series) -> float:
    """The geometric mean of the positive values in `x`, `nan` if there are none."""
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def lighten(color: str, amount: float = 0.55) -> tuple:
    """Blend `color` toward white by `amount`, for a SIMD series' line color."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def density_cmap(color) -> LinearSegmentedColormap:
    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list(
        "density", [(r, g, b, 0.0), (r, g, b, 1.0)]
    )


def density_hexbin(
    ax, x: pd.Series, y: pd.Series, color, extent, log: bool = True
) -> None:
    """A density heatmap of `(x, y)` in `color`, faded by point density so overlapping series
    stay visually distinguishable rather than fully occluding each other.

    `extent` is always given in raw data units (matching `ax.set_xlim`/`set_ylim`); when
    `log=True` it's converted to the log10-space `hexbin` itself expects for a log `xscale`/
    `yscale` (passing raw units there silently produces nonsensical bins spanning many decades
    beyond the actual data, an easy mistake since every other matplotlib API takes raw units).
    """
    scale = "log" if log else "linear"
    if log:
        x_lo, x_hi, y_lo, y_hi = extent
        extent = (np.log10(x_lo), np.log10(x_hi), np.log10(y_lo), np.log10(y_hi))
    ax.hexbin(
        x,
        y,
        gridsize=22,
        cmap=density_cmap(color),
        mincnt=1,
        linewidths=0.0,
        bins="log",
        xscale=scale,
        yscale=scale,
        extent=extent,
    )


# Number of quantile bins to plot.
N_QUANTILE_BINS = 10


def binned_line(
    ax,
    x: pd.Series,
    y: pd.Series,
    color,
    linestyle: str = "-",
    annotate: bool = True,
    label: str | None = None,
) -> None:
    """Plot a quantile-binned geometric-mean trend line of `y` against `x`, annotated with each
    bin's sample count."""
    n_bins = min(N_QUANTILE_BINS, x.nunique())
    if n_bins < 2:
        return
    bin_id = pd.qcut(x, n_bins, duplicates="drop")
    grouped = y.groupby(bin_id, observed=True)
    xs = x.groupby(bin_id, observed=True).median()
    ys = grouped.apply(geomean)
    counts = grouped.size()
    (line,) = ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2.5,
        linestyle=linestyle,
        marker="o",
        markersize=4,
        label=label,
    )
    # add white halo for contrast
    line.set_path_effects(
        [patheffects.Stroke(linewidth=4.5, foreground="white"), patheffects.Normal()]
    )
    if annotate:
        for bx, by, n in zip(xs, ys, counts):
            ax.annotate(
                f"n={n}",
                (bx, by),
                textcoords="offset points",
                xytext=(0, 6),
                fontsize=6,
                ha="center",
            )
