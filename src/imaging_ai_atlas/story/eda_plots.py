from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Importing plots also selects the headless Agg backend (see the note there).
from imaging_ai_atlas.story.plots import (
    BLUE,
    GREY,
    LINE_COLORS,
    save_figure,
    stack_end_labels,
)

_INK = "#333333"


def _style(ax: Axes, value_axis: Literal["x", "y"]) -> None:
    """Recessive frame: no top/right spines and a light grid on the value axis only. On
    horizontal charts (value axis x) the category axis loses its spine and ticks too,
    since the row labels already mark each row.
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GREY)
    ax.tick_params(colors=_INK, labelsize=9)
    ax.grid(axis=value_axis, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    if value_axis == "x":
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def _finish(
    fig: Figure,
    ax: Axes,
    title: str,
    value_label: str,
    value_axis: Literal["x", "y"],
    save_path: Path,
) -> None:
    """Left-aligned title, the value-axis label, then save."""
    set_label = ax.set_xlabel if value_axis == "x" else ax.set_ylabel
    set_label(value_label, color=_INK)
    ax.set_title(title, loc="left", color=_INK)
    save_figure(fig, save_path)


def plot_papers_per_year(counts: pd.Series, save_path: Path, last_year_partial: bool) -> None:
    """Bar per year. With `last_year_partial` (the snapshot ends partway through its last
    year), that bar is drawn in grey and labelled so its lower count doesn't read as a
    decline.
    """
    years = [int(y) for y in counts.index]
    values = [int(v) for v in counts.to_numpy()]
    colors = [BLUE] * len(years)
    if last_year_partial:
        colors[-1] = GREY

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(years, values, color=colors, width=0.8)
    _style(ax, value_axis="y")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")

    peak = max(values)
    ax.annotate(
        f"{peak:,}",
        (years[values.index(peak)], peak),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=_INK,
    )
    if last_year_partial:
        # Written vertically inside its own bar: above it, it would collide with the
        # label of the (usually similar-height) bar next to it.
        ax.annotate(
            "partial year",
            (years[-1], values[-1]),
            xytext=(0, -6),
            textcoords="offset points",
            ha="center",
            va="top",
            rotation=90,
            fontsize=8,
            color=_INK,
        )

    _finish(fig, ax, "Medical-imaging AI papers on arXiv per year", "Papers", "y", save_path)


def plot_corpus_funnel(stages: list[tuple[str, int]], save_path: Path) -> None:
    """One dot per pipeline stage on a log axis, labelled with its count and the share it
    keeps of the stage before. Dots, not bars: on a log scale bar length isn't
    proportional to the count, and on a linear one the later stages would be invisible.
    """
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    rows = list(range(len(stages)))[::-1]  # first stage at the top

    fig, ax = plt.subplots(figsize=(9, 0.9 * len(stages) + 1))
    ax.scatter(values, rows, s=90, color=BLUE, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(min(values) / 3, max(values) * 4)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.set_yticks(rows, labels)
    _style(ax, value_axis="x")
    ax.tick_params(axis="y", labelsize=10)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")

    for i, (value, row) in enumerate(zip(values, rows, strict=True)):
        note = f"{value:,}"
        if i > 0:
            note += f"   ({value / values[i - 1]:.1%} of the stage above)"
        ax.annotate(
            note,
            (value, row),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=_INK,
        )

    _finish(fig, ax, "How the corpus narrows, stage by stage", "Papers (log scale)", "x", save_path)


def plot_category_share(share: pd.DataFrame, save_path: Path) -> None:
    """One line per category column of a year x category share table, labelled in the
    right margin instead of a legend.
    """
    years = [int(y) for y in share.index]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_ylim(0, 1)

    entries: list[tuple[str, str, float, float]] = []
    for i, category in enumerate(share.columns):
        values = [float(v) for v in share[category].to_numpy()]
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ax.plot(years, values, color=color, linewidth=2)
        entries.append((color, str(category), years[-1], values[-1]))

    label_x = years[-1] + 1
    ax.set_xlim(right=label_x + 2)
    # Short chart, so labels need a bigger share of the y-range to clear each other.
    stack_end_labels(ax, entries, label_x, min_gap_fraction=1 / 20)

    _style(ax, value_axis="y")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _finish(
        fig,
        ax,
        "Which arXiv categories the papers are filed under",
        "Share of that year's papers",
        "y",
        save_path,
    )


def plot_largest_topics(ranked: pd.DataFrame, save_path: Path) -> None:
    """Horizontal bars, largest theme at the top, each labelled with its paper count."""
    labels = ranked["label"].tolist()[::-1]
    sizes = ranked["size"].tolist()[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.38 * len(labels) + 1.2))
    bars = ax.barh(labels, sizes, color=BLUE, height=0.7)
    ax.bar_label(bars, labels=[f"{s:,}" for s in sizes], padding=4, fontsize=8, color=_INK)
    _style(ax, value_axis="x")
    ax.set_xlim(right=max(sizes) * 1.12)
    _finish(fig, ax, f"The {len(labels)} largest research themes", "Papers", "x", save_path)
