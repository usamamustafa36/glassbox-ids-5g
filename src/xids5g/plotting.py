"""Figure helpers.

All figures are static PNGs destined for the report and the
hand-out, so they commit to a single light appearance. The categorical palette
is a validated colour-blind-safe order used in fixed slot order (never cycled);
bars carry direct value labels, which also satisfies the contrast relief rule
for the lighter slots. Grid and axes are hairlines, one shade off the surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")           # headless: identical output on Windows and Linux
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# -- design tokens ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8b8a85"
GRID = "#e6e5e1"

#: Fixed categorical order. Never cycled, never assigned by rank.
SERIES: list[str] = [
    "#2a78d6",   # 1 blue
    "#eb6834",   # 2 orange
    "#1baf7a",   # 3 aqua
    "#eda100",   # 4 yellow
    "#e87ba4",   # 5 magenta
    "#008300",   # 6 green
    "#4a3aa7",   # 7 violet
    "#e34948",   # 8 red
]

#: Single-hue sequential ramp (light -> dark) for magnitude.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fe", "#cde2fb", "#9ec5f4", "#5598e7",
                 "#2a78d6", "#184f95", "#0d366b"])

#: Diverging: warm/cool poles with a neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_br", ["#104281", "#5598e7", "#cde2fb", "#f0efec",
               "#f6c3c3", "#e34948", "#8f1f1f"])

STATUS = {"good": "#008300", "warning": "#eda100", "critical": "#e34948"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


apply_style()


def _finish(fig: plt.Figure, path: Path, subtitle: str | None = None,
            subtitle_y: float = -0.02) -> Path:
    if subtitle:
        # Wrap by hand: figure-space text does not reflow on its own.
        fig.text(0.01, subtitle_y, _wrap(subtitle, 118), fontsize=8,
                 color=INK_MUTED, ha="left", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width)) or text


def _despine(ax: plt.Axes, keep_x: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    if not keep_x:
        ax.spines["bottom"].set_visible(False)


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------
def barh(labels: Sequence[str], values: Sequence[float], path: Path, *,
         title: str, xlabel: str = "", subtitle: str | None = None,
         color: str | None = None, fmt: str = "{:.3f}",
         highlight: Sequence[str] = (), figsize=None) -> Path:
    """Single-series horizontal bars, one colour for every bar."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=figsize or (8, max(2.4, 0.42 * n + 1.3)))
    base = color or SERIES[0]
    colors = [STATUS["critical"] if l in highlight else base for l in labels]
    y = np.arange(n)
    ax.barh(y, values, height=0.62, color=colors, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", zorder=0)
    _despine(ax)
    span = max(values) - min(0, min(values)) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + span * 0.012, yi, fmt.format(v), va="center", ha="left",
                fontsize=8.5, color=INK_2)
    ax.set_xlim(min(0, float(np.min(values)) * 1.1),
                float(np.max(values)) * 1.18 if len(values) else 1)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    return _finish(fig, path, subtitle)


def grouped_bars(categories: Sequence[str], series: dict[str, Sequence[float]],
                 path: Path, *, title: str, ylabel: str = "",
                 subtitle: str | None = None, fmt: str | None = "{:.2f}",
                 ylim: tuple[float, float] | None = None,
                 figsize=None, rotate: int = 0) -> Path:
    """Grouped vertical bars. Series take categorical slots in fixed order."""
    names = list(series)
    n_s, n_c = len(names), len(categories)
    width = 0.8 / n_s
    fig, ax = plt.subplots(figsize=figsize or (max(7.0, 1.15 * n_c + 2), 4.4))
    x = np.arange(n_c)
    for i, name in enumerate(names):
        off = (i - (n_s - 1) / 2) * width
        vals = np.asarray(series[name], dtype=float)
        # 2px surface gap between adjacent bars, not a border
        ax.bar(x + off, vals, width * 0.88, label=name, color=SERIES[i % 8],
               zorder=3)
        if fmt and n_c * n_s <= 40:
            for xi, v in zip(x + off, vals):
                if np.isfinite(v):
                    ax.text(xi, v, " " + fmt.format(v), ha="center",
                            va="bottom", fontsize=7.2, color=INK_2,
                            rotation=90 if n_s > 3 else 0)
    ax.set_xticks(x, categories, rotation=rotate,
                  ha="right" if rotate else "center")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", zorder=0)
    _despine(ax)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, loc="left")
    sub_y = -0.02
    if n_s >= 2:
        anchor = -0.16 if rotate else -0.10
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, anchor),
                  ncol=min(n_s, 4))
        sub_y = -0.10 if rotate else -0.06
    return _finish(fig, path, subtitle, subtitle_y=sub_y)


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------
def lines(x: Sequence[float], series: dict[str, Sequence[float]], path: Path, *,
          title: str, xlabel: str = "", ylabel: str = "",
          subtitle: str | None = None, direct_label: bool = True,
          figsize=None, dashed: Sequence[str] = ()) -> Path:
    fig, ax = plt.subplots(figsize=figsize or (7.6, 4.4))
    names = list(series)
    for i, name in enumerate(names):
        ax.plot(x, series[name], label=name, color=SERIES[i % 8],
                linestyle="--" if name in dashed else "-",
                marker="o" if len(x) <= 12 else None,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        if direct_label and len(names) <= 4:
            ax.annotate(name, (x[-1], list(series[name])[-1]),
                        textcoords="offset points", xytext=(6, 0),
                        fontsize=8.5, color=SERIES[i % 8], va="center")
    _despine(ax)
    ax.grid(zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if len(names) >= 2:
        ax.legend(loc="best", ncol=min(len(names), 3))
    if direct_label and len(names) <= 4:
        ax.margins(x=0.12)
    return _finish(fig, path, subtitle)


# --------------------------------------------------------------------------
# Matrices
# --------------------------------------------------------------------------
def heatmap(matrix: np.ndarray, row_labels: Sequence[str],
            col_labels: Sequence[str], path: Path, *, title: str,
            subtitle: str | None = None, fmt: str = "{:.0f}",
            cmap=None, diverging: bool = False, cbar_label: str = "",
            figsize=None, annotate: bool = True) -> Path:
    n_r, n_c = matrix.shape
    fig, ax = plt.subplots(figsize=figsize or (max(6.0, 0.78 * n_c + 3.2),
                                               max(4.6, 0.62 * n_r + 2.4)))
    if diverging:
        lim = float(np.nanmax(np.abs(matrix))) or 1.0
        im = ax.imshow(matrix, cmap=cmap or DIVERGING, vmin=-lim, vmax=lim)
    else:
        im = ax.imshow(matrix, cmap=cmap or SEQ_BLUE)
    ax.set_xticks(np.arange(n_c), col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_r), row_labels)
    ax.grid(visible=False)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2, visible=True)
    ax.tick_params(which="minor", length=0)
    if annotate:
        vmax = float(np.nanmax(np.abs(matrix))) or 1.0
        for i in range(n_r):
            for j in range(n_c):
                v = matrix[i, j]
                if not np.isfinite(v):
                    continue
                shade = abs(v) / vmax
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=7.8,
                        color="#ffffff" if shade > 0.55 else INK_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=GRID)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=9, color=INK_2)
    ax.set_title(title, loc="left")
    return _finish(fig, path, subtitle, subtitle_y=-0.09)


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------
def boxes(groups: dict[str, Sequence[float]], path: Path, *, title: str,
          ylabel: str = "", subtitle: str | None = None, figsize=None) -> Path:
    names = list(groups)
    fig, ax = plt.subplots(figsize=figsize or (max(5.6, 1.5 * len(names) + 2), 4.4))
    data = [np.asarray(groups[n], dtype=float) for n in names]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color=INK, linewidth=1.6),
                    whiskerprops=dict(color=INK_MUTED, linewidth=1.0),
                    capprops=dict(color=INK_MUTED, linewidth=1.0),
                    flierprops=dict(marker="o", markersize=4,
                                    markerfacecolor=INK_MUTED,
                                    markeredgecolor=SURFACE, alpha=0.7))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(SERIES[i % 8])
        box.set_alpha(0.85)
        box.set_edgecolor(SURFACE)
        box.set_linewidth(2)
    for i, d in enumerate(data):                 # jittered raw points
        xj = np.random.default_rng(0).normal(i + 1, 0.045, size=len(d))
        ax.plot(xj, d, "o", ms=3.5, color=INK, alpha=0.45,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
    ax.set_xticks(np.arange(1, len(names) + 1), names)
    ax.grid(axis="x", visible=False)
    _despine(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    return _finish(fig, path, subtitle)


def shape_function(x: Sequence[float], y: Sequence[float], path: Path, *,
                   title: str, xlabel: str, subtitle: str | None = None,
                   density: Sequence[float] | None = None,
                   bin_edges: Sequence[float] | None = None,
                   figsize=None) -> Path:
    """An EBM shape function: contribution to the log-odds vs feature value."""
    fig, ax = plt.subplots(figsize=figsize or (7.0, 4.0))
    ax.axhline(0, color=INK_MUTED, linewidth=0.9, zorder=2)
    ax.step(x, y, where="post", color=SERIES[0], zorder=4)
    ax.fill_between(x, 0, y, step="post", color=SERIES[0], alpha=0.13, zorder=3)
    _despine(ax)
    ax.grid(zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("contribution to log-odds of 'attack'")
    ax.set_title(title, loc="left")
    if density is not None and bin_edges is not None and len(density):
        ax2 = ax.inset_axes([0, -0.28, 1, 0.16], sharex=ax)
        ax2.bar(bin_edges[:-1], density, width=np.diff(bin_edges),
                align="edge", color=INK_MUTED, alpha=0.5)
        ax2.set_yticks([])
        ax2.grid(visible=False)
        for s in ax2.spines.values():
            s.set_visible(False)
        ax2.set_xlabel(f"{xlabel}  (distribution of the data)", fontsize=8.5)
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
        return _finish(fig, path, subtitle, subtitle_y=-0.36)
    return _finish(fig, path, subtitle)


def categorical_shape(labels: Sequence[str], values: Sequence[float], path: Path,
                      *, title: str, xlabel: str, subtitle: str | None = None,
                      figsize=None) -> Path:
    """An EBM shape function over a nominal feature: one bar per category.

    Direction carries the meaning here, so the bars are coloured by sign from
    the diverging pair rather than by category identity.
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=figsize or (max(6.0, 0.9 * n + 2.4), 4.2))
    vals = np.asarray(values, dtype=float)
    colors = ["#e34948" if v > 0 else "#2a78d6" for v in vals]
    ax.bar(np.arange(n), vals, width=0.62, color=colors, zorder=3)
    ax.axhline(0, color=INK_MUTED, linewidth=0.9, zorder=4)
    ax.set_xticks(np.arange(n), labels, rotation=30, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color=INK_2)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", zorder=0)
    _despine(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("contribution to log-odds of 'attack'")
    ax.set_title(title, loc="left")
    ax.margins(y=0.18)
    return _finish(fig, path, subtitle, subtitle_y=-0.14)


def scatter(x: Sequence[float], y: Sequence[float], path: Path, *, title: str,
            xlabel: str, ylabel: str, subtitle: str | None = None,
            labels: Sequence[str] | None = None, figsize=None,
            identity_line: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=figsize or (6.2, 5.4))
    if identity_line:
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1.0, zorder=2)
    ax.scatter(x, y, s=48, color=SERIES[0], edgecolor=SURFACE, linewidth=1.6,
               zorder=4, alpha=0.9)
    if labels is not None:
        for xi, yi, li in zip(x, y, labels):
            ax.annotate(li, (xi, yi), textcoords="offset points",
                        xytext=(7, 3), fontsize=7.5, color=INK_2)
    _despine(ax)
    ax.grid(zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    return _finish(fig, path, subtitle)


def stat_tiles(tiles: list[tuple[str, str, str]], path: Path, *, title: str,
               subtitle: str | None = None) -> Path:
    """A row of headline numbers - used where a chart would be one bar."""
    n = len(tiles)
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n, 2.25))
    if n == 1:
        axes = [axes]
    for ax, (label, value, caption) in zip(axes, tiles):
        ax.axis("off")
        ax.text(0.5, 0.80, label, ha="center", va="center", fontsize=9.5,
                color=INK_2, transform=ax.transAxes)
        ax.text(0.5, 0.48, value, ha="center", va="center", fontsize=25,
                fontweight="bold", color=SERIES[0], transform=ax.transAxes)
        ax.text(0.5, 0.16, caption, ha="center", va="center", fontsize=8,
                color=INK_MUTED, transform=ax.transAxes, wrap=True)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold")
    return _finish(fig, path, subtitle)
