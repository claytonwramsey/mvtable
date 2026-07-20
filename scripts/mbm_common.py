"""Shared helpers for the `mbm_bench` result-plotting scripts."""

import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects
from matplotlib.category import StrCategoryLocator
from matplotlib.colors import to_rgb
from matplotlib.ticker import FixedLocator, FuncFormatter

# Locator types that mean "this axis has category positions, not numeric ones": `FixedLocator`
# from an explicit `ax.set_xticks(positions, labels)` call (e.g. plot_baxter_solve_time.py's
# structure names), and `StrCategoryLocator` from plotting directly against a string column (e.g.
# plot_mbm_plan.py's `x="robot"`, handled entirely by seaborn/matplotlib's categorical machinery).
CATEGORICAL_LOCATORS = (FixedLocator, StrCategoryLocator)

# Human-readable robot names, for any axis/legend that would otherwise show the raw MotionBenchMaker
# dataset key (e.g. "ur5").
ROBOT_LABELS = {
    "panda": "Panda",
    "ur5": "UR5",
    "fetch": "Fetch",
    "baxter": "Baxter",
}

# Canonical structure -> color, shared by every script that plots more than one structure
# (`plot_mbm.py`, `plot_baxter_solve_time.py`, `plot_mbm_plan.py`). Keyed by the `mvtable`-style
# structure names `mbm_bench` writes; scripts reading `mbm_plan_results.csv`'s slightly different
# names (e.g. `mvtable_cpp` instead of `mvt_cpp`) look up the same underlying color by the
# `mbm_bench` key and re-key it locally.
#
# `mvtable`/`mvtable_mutable` are the site's own `--bh-blue`/`--bh-red` (see `main.css`) - the two
# most prominent series get the two colors a reader already associates with this site. The other
# three keep their original Okabe-Ito colorblind-safe values rather than being reinvented in a
# site-branded color: Okabe-Ito was already vivid and well-tested, and re-deriving a whole new set
# of secondary colors (an earlier version of this palette tried pulling from the syntax-highlight
# palette in `code-light.css`) produced muddier, less legible lines for no real benefit. Checked
# with `colorspacious` (simulated deuteranomaly/protanomaly/tritanomaly, full severity): every
# pair below clears deltaE >= 17, edging out Okabe-Ito's own worst-case pairwise separation (~16).
STRUCTURE_COLORS = {
    "mvtable": "#3F51B5",  # site --bh-blue
    "mvtable_mutable": "#E03C31",  # site --bh-red
    "capt": "#009E73",  # Okabe-Ito bluish green
    "kiddo": "#E69F00",  # Okabe-Ito orange
    "mvt_cpp": "#CC79A7",  # Okabe-Ito reddish purple
}

# Canonical robot -> color, used only by `plot_voxel_width_sweep.py`. Deliberately a *different*
# set of hues from STRUCTURE_COLORS, not a reuse of any of them: robots and collision-checking
# structures are unrelated categorical dimensions that never appear in the same figure, but a
# reader who's learned "blue = mvtable" from the throughput figures shouldn't then see a similar
# blue for "panda" here and wonder if it's related. Checked the same way as STRUCTURE_COLORS:
# every pair here clears CVD deltaE >= 26 internally, and every color clears normal-vision deltaE
# >= 27 from every STRUCTURE_COLORS entry (no near-duplicate hue to cause that habit confusion).
ROBOT_COLORS = {
    "panda": "#00A8C6",  # cyan
    "ur5": "#8B5A2B",  # brown
    "fetch": "#5B2873",  # violet
    "baxter": "#AEA02C",  # gold/olive
}

# Negative labelpad on every panel's y-axis label, pulled in from matplotlib's default (4pt) so
# the label sits closer to the tick numbers rather than leaving a wide gap of unused margin to its
# left - `fig.tight_layout()` shrinks the figure's left margin to match, so the freed-up space
# goes to the plot area instead of sitting blank.
YLABEL_PAD = -8


def wrap_tick_label(label: str) -> str:
    """Break `label` one word per line (keeping a parenthesized suffix like "(SIMD x8)" together
    as its own line), e.g. "Mutable MVT (SIMD x8)" -> "Mutable\nMVT\n(SIMD x8)", so labels stay
    narrow enough to sit horizontally under their violin without colliding with their neighbors -
    unlike a rotated label, a horizontal one can't lean on diagonal clearance for width."""
    base, _, paren = label.partition(" (")
    lines = base.split(" ")
    if paren:
        lines.append("(" + paren)
    return "\n".join(lines)


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


def save_figure(fig, out: pathlib.Path, crop: bool = False) -> None:
    """Write `fig` to `out` (SVG, transparent background so it renders on dark and light
    pages alike) and to a same-named `.png` (opaque, for contexts that need a raster).

    `crop=True` trims the figure to the bounding box of its artists (no outer margin), for
    embedding on a page that supplies its own layout/padding rather than relying on matplotlib's
    figure-sized canvas."""
    out.parent.mkdir(exist_ok=True)
    save_kwargs = {"transparent": True}
    if crop:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = 0
    fig.savefig(out, **save_kwargs)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")


def _is_categorical(axis) -> bool:
    """Whether `axis` has category positions rather than ticks placed automatically by a numeric
    locator like `LogLocator` - covers both the robot/structure names set via an explicit
    `ax.set_xticks(positions, labels)` call (`FixedLocator`) and ones plotted directly against a
    string column, e.g. `x="robot"` (`StrCategoryLocator`); see `CATEGORICAL_LOCATORS`.

    Either way, tick labels are looked up by each tick's position in a fixed list, not derived
    from its numeric value (for the `FixedLocator` case, matplotlib installs this via a
    `FuncFormatter` closing over a `{position: label}` dict, not a `FixedFormatter`, so checking
    the locator is the reliable signal, not the formatter type). Dropping or adding a tick would
    shift every later label onto the wrong position instead of just adding/removing one, so both
    `_clip_ticks` and `_label_endpoints` skip these axes entirely.

    Must be checked once, before either of those two functions runs: both call `axis.set_ticks`,
    which itself installs a `FixedLocator` - so checking again *between* them would see their own
    prior mutation and misidentify a numeric axis as categorical too."""
    return isinstance(axis.get_major_locator(), CATEGORICAL_LOCATORS)


def _clip_ticks(axis, lo: float, hi: float) -> None:
    """Drop major/minor tick marks outside `[lo, hi]`, so none floats past the spine end that
    `trim_spines_to_data` just set. Caller must already have excluded categorical axes."""
    axis.set_ticks([t for t in axis.get_majorticklocs() if lo <= t <= hi], minor=False)
    axis.set_ticks([t for t in axis.get_minorticklocs() if lo <= t <= hi], minor=True)


def _format_endpoint(value: float, sig: int = 3) -> str:
    """`value` to `sig` significant figures, plain decimal notation, no trailing zeros - for
    labeling a range-frame's exact data endpoint, which (unlike the regular round-number ticks)
    is essentially never itself a round number."""
    if value == 0:
        return "0"
    decimals = max(sig - 1 - int(np.floor(np.log10(abs(value)))), 0)
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


# A regular round-number tick within this fraction of the total visible span (in display
# coordinates - log10-space on a log axis) of an endpoint gets dropped in `_label_endpoints`,
# since its label would otherwise sit close enough to the endpoint's own label to visually
# collide with it (e.g. a data max of 15140 landing right next to an existing "10^4" tick).
ENDPOINT_PROXIMITY_FRAC = 0.06


def _label_endpoints(axis, lo: float, hi: float) -> None:
    """Add tick labels at the exact `lo`/`hi` data endpoints (Tufte range-frame style), alongside
    whatever round-number ticks `_clip_ticks` kept, so the spine's ends read their real extreme
    values rather than only the nearest round tick. Drops any regular tick that would land close
    enough to crowd an endpoint's label (see `ENDPOINT_PROXIMITY_FRAC`).

    Caller must already have excluded categorical axes; see `_is_categorical`."""
    base_formatter = axis.get_major_formatter()
    to_display = np.log10 if axis.get_scale() == "log" else (lambda v: v)

    span = to_display(hi) - to_display(lo)
    threshold = ENDPOINT_PROXIMITY_FRAC * span
    ticks = [
        t
        for t in axis.get_majorticklocs()
        if not any(abs(to_display(t) - to_display(v)) < threshold for v in (lo, hi))
    ]
    axis.set_ticks(sorted(ticks + [lo, hi]), minor=False)

    # `base_formatter` (typically a `ScalarFormatter`) only knows the tick locations passed to its
    # own `set_locs` - normally done by the draw cycle for whichever formatter is currently
    # installed. Since it's about to be wrapped in `relabel` below instead of installed directly,
    # that call would never happen, and a `ScalarFormatter` renders every non-endpoint tick as ''
    # when it hasn't seen any locations (it can't compute its shared order-of-magnitude/offset).
    # Pass only the regular (non-endpoint) ticks: `lo`/`hi` are irregular data values that never
    # go through `base_formatter` anyway (see `relabel` below), and including them would skew its
    # shared decimal precision for every other tick too (e.g. "0.10" instead of "0.1" just because
    # an endpoint like 0.05 needs two decimal places).
    base_formatter.set_locs(ticks)

    def relabel(x, pos=None):
        if any(np.isclose(x, v, rtol=1e-9) for v in (lo, hi)):
            return _format_endpoint(x)
        return base_formatter(x, pos)

    axis.set_major_formatter(FuncFormatter(relabel))


def trim_spines_to_data(ax) -> None:
    """Trim the left/bottom spines to the tight bounding box of the plotted data (a Tufte
    range-frame) instead of the full, margin-padded axis limits, so each spine's own extent shows
    the data's range rather than an arbitrary rectangle. Also drops ticks beyond that range (which
    would otherwise float past the now-shorter spine) and labels the exact endpoints. Call after
    all of `ax`'s data is plotted (and after `sns.despine`, though order with that doesn't
    actually matter)."""
    (x_lo, x_hi), (y_lo, y_hi) = ax.dataLim.intervalx, ax.dataLim.intervaly
    if np.isfinite(x_lo) and np.isfinite(x_hi):
        ax.spines["bottom"].set_bounds(x_lo, x_hi)
        if not _is_categorical(ax.xaxis):
            _clip_ticks(ax.xaxis, x_lo, x_hi)
            _label_endpoints(ax.xaxis, x_lo, x_hi)
    if np.isfinite(y_lo) and np.isfinite(y_hi):
        ax.spines["left"].set_bounds(y_lo, y_hi)
        if not _is_categorical(ax.yaxis):
            _clip_ticks(ax.yaxis, y_lo, y_hi)
            _label_endpoints(ax.yaxis, y_lo, y_hi)


def finish_single_panel(ax, xlabel: str, yscale: str = "log") -> None:
    """Apply the shared single-panel look used across every plot in this directory: linear x,
    `sns.despine`, and a Tufte range-frame via `trim_spines_to_data`.

    `yscale` defaults to `"log"` since every time/memory metric plotted in this directory spans a
    comparable multi-order-of-magnitude range that a linear scale would crush into a sliver at the
    small end - pass `"linear"` for a metric that doesn't (e.g. `plot_voxel_width_sweep.py`'s
    query time barely varies across its swept range, well under one decade; log-scale ticks over
    that narrow a span render as wide "n×10^k" labels rather than plain numbers, which collide
    with `YLABEL_PAD`'s tightened label margin)."""
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    sns.despine(ax=ax)
    trim_spines_to_data(ax)


def legend_order(
    handles, labels, pin_first: str | None = None, fixed_order: list[str] | None = None
):
    """Reorder legend entries to match the plotted lines' relative vertical order at the left
    edge of the plot (where a left-to-right reader looks first), so e.g. the highest series reads
    first/top in the legend rather than in a fixed structure order.

    `pin_first`, if given, forces that label to the top regardless of its left-edge position -
    useful when a series starts close to the pack at the smallest x value but consistently leads
    or trails for nearly the whole plot, so top-of-legend is the more honest read than "whatever's
    highest at x=0".

    `fixed_order`, if given, replaces height-based ordering entirely with this explicit sequence
    of labels - useful when the intended reading order doesn't track well enough with the lines'
    relative height at any single x position to trust automatic ordering. Mutually exclusive with
    `pin_first`."""
    if fixed_order is not None:
        order = sorted(
            range(len(handles)),
            key=lambda i: (
                fixed_order.index(labels[i])
                if labels[i] in fixed_order
                else len(fixed_order)
            ),
        )
    else:
        order = sorted(
            range(len(handles)), key=lambda i: handles[i].get_ydata()[0], reverse=True
        )
        if pin_first is not None:
            order = sorted(order, key=lambda i: labels[i] != pin_first)

    return [handles[i] for i in order], [labels[i] for i in order]


def style_legend(ax, handles, labels, **kwargs):
    """Draw a legend with the shared look used across every plot in this directory: no frame,
    tightened handle/label spacing, small font, and line swatches thinned to 1.6pt regardless of
    the plotted lines' own width (full-thickness swatches read as a stack of fat bars once a
    legend has more than a couple of rows). Extra `kwargs` (e.g. `loc`, `ncol`, `bbox_to_anchor`)
    are passed through to `ax.legend`."""
    legend = ax.legend(
        handles,
        labels,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.5,
        labelspacing=0.35,
        fontsize=9,
        **kwargs,
    )
    for handle in legend.legend_handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(1.6)
    return legend


def geomean(x: pd.Series) -> float:
    """The geometric mean of the positive values in `x`, `nan` if there are none."""
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def lighten(color: str, amount: float = 0.55) -> tuple:
    """Blend `color` toward white by `amount`, for a SIMD series' line color."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# Number of linear bins to plot.
N_LINEAR_BINS = 20


def binned_line(
    ax,
    x: pd.Series,
    y: pd.Series,
    color,
    linestyle: str = "-",
    annotate: bool = True,
    label: str | None = None,
) -> None:
    """Plot a linearly-binned geometric-mean trend line of `y` against `x` (equal-width bins
    spanning `x`'s range, matching the linear x-axis these panels use), annotated with each bin's
    sample count."""
    n_bins = min(N_LINEAR_BINS, x.nunique())
    if n_bins < 2:
        return
    bin_id = pd.cut(x, n_bins, duplicates="drop")
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
        # marker="o",
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
