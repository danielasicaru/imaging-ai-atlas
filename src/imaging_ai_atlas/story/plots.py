from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Figures only ever get saved to a file, never shown interactively - force the
# non-interactive Agg backend so this works headless (CI, no Tk/X11) without needing a
# display, rather than relying on whatever backend matplotlib happens to autodetect.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Visualization rule for this project: one highlight colour family + grey for everything
# else, direct labels instead of a legend (a legend forces the reader to look back and
# forth; a label placed right at the data point doesn't), no interactivity in static
# figures since there's no hover affordance to add information through.
_HIGHLIGHT_CMAP = plt.get_cmap("tab10")
_GREY = "#bbbbbb"


def _spread_labels(
    positions: list[tuple[float, float]], min_gap: float, x_gap: float
) -> list[tuple[float, float]]:
    """Nudge each label downward, in placement order, until it's at least `min_gap` away
    (vertically) from every previously placed label within `x_gap` horizontally - a
    lightweight stand-in for a full label-layout solver (not worth a new dependency for
    ~10 labels). Dense topic clusters sit close together on the map, so unmoved labels
    would otherwise overlap into unreadable text.
    """
    placed: list[tuple[float, float]] = []
    for x, y in positions:
        new_y = y
        moved = True
        while moved:
            moved = False
            for px, py in placed:
                if abs(x - px) < x_gap and abs(new_y - py) < min_gap:
                    new_y += min_gap
                    moved = True
        placed.append((x, new_y))
    return placed


def plot_topic_map(
    coords: np.ndarray,
    topics: list[int],
    topic_labels: dict[int, str],
    save_path: Path,
    n_highlighted: int = 10,
) -> None:
    """2D scatter of every document, colored by its topic for the `n_highlighted` largest
    topics (direct-labeled near each topic's centroid, nudged apart where clusters sit
    close together) and grey for everything else (smaller topics and the outlier bucket) -
    highlighting all ~190 topics at once would be unreadable noise, not a story.
    """
    topics_arr = np.asarray(topics)
    sizes = pd.Series(topics).value_counts()
    highlighted = [t for t in sizes.index if t != -1][:n_highlighted]

    fig, ax = plt.subplots(figsize=(12, 10))
    other_mask = ~np.isin(topics_arr, highlighted)
    ax.scatter(coords[other_mask, 0], coords[other_mask, 1], c=_GREY, s=4, alpha=0.4, linewidths=0)

    centroids = []
    for topic_id in highlighted:
        mask = topics_arr == topic_id
        cx, cy = coords[mask].mean(axis=0)
        centroids.append((float(cx), float(cy)))

    x_span = coords[:, 0].max() - coords[:, 0].min()
    y_span = coords[:, 1].max() - coords[:, 1].min()
    label_positions = _spread_labels(centroids, min_gap=y_span / 30, x_gap=x_span / 6)

    for i, topic_id in enumerate(highlighted):
        mask = topics_arr == topic_id
        color = _HIGHLIGHT_CMAP(i % _HIGHLIGHT_CMAP.N)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], s=6, alpha=0.8, linewidths=0)
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        cx, cy = centroids[i]
        lx, ly = label_positions[i]
        needs_leader = (lx, ly) != (cx, cy)
        ax.annotate(
            label,
            xy=(cx, cy),
            xytext=(lx, ly),
            fontsize=8,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            arrowprops=(
                {"arrowstyle": "-", "color": color, "linewidth": 0.5, "alpha": 0.6}
                if needs_leader
                else None
            ),
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Medical-imaging AI research: topic map")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_topic_dynamics(
    share: pd.DataFrame,
    topic_labels: dict[int, str],
    save_path: Path,
    n_highlighted: int = 8,
    min_year_total: int = 20,
) -> None:
    """One line per highlighted topic (its share of the corpus per year), with every
    other topic summed into a single grey "other topics" line. Years with fewer than
    `min_year_total` non-outlier papers are dropped - with only a handful of papers, one
    extra document swings a topic's "share" wildly, which would read as a dramatic trend
    that's really just small-sample noise (arXiv's medical-imaging-AI output was tiny
    before the deep-learning era). Labels are stacked in the right margin, sorted by each
    line's final share and connected to its actual last point with a thin leader line -
    convergent lines would otherwise pile all their direct labels on top of each other.
    """
    filtered = share[share["year_total"] >= min_year_total]
    columns = filtered[["year", "topic_id", "count", "share"]]
    records: list[dict[str, int | float]] = columns.to_dict("records")  # type: ignore[assignment]

    total_by_topic: dict[int, float] = {}
    for row in records:
        topic_id = int(row["topic_id"])
        total_by_topic[topic_id] = total_by_topic.get(topic_id, 0.0) + row["count"]
    highlighted = sorted(total_by_topic, key=lambda t: total_by_topic[t], reverse=True)[
        :n_highlighted
    ]

    other_by_year: dict[int, float] = {}
    series_by_topic: dict[int, list[tuple[int, float]]] = {t: [] for t in highlighted}
    for row in records:
        topic_id = int(row["topic_id"])
        year = int(row["year"])
        if topic_id in series_by_topic:
            series_by_topic[topic_id].append((year, row["share"]))
        else:
            other_by_year[year] = other_by_year.get(year, 0.0) + row["share"]

    fig, ax = plt.subplots(figsize=(12, 6))

    other_years = sorted(other_by_year)
    other_shares = [other_by_year[y] for y in other_years]
    ax.plot(other_years, other_shares, color=_GREY, linewidth=1.5)

    entries: list[tuple[str | tuple[float, ...], str, float, float]] = []
    if other_years:
        entries.append((_GREY, "other topics", other_years[-1], other_shares[-1]))

    all_years = list(other_years)
    for i, topic_id in enumerate(highlighted):
        points = sorted(series_by_topic[topic_id])
        if not points:
            continue
        years = [p[0] for p in points]
        shares = [p[1] for p in points]
        all_years.extend(years)
        color = _HIGHLIGHT_CMAP(i % _HIGHLIGHT_CMAP.N)
        ax.plot(years, shares, color=color, linewidth=2)
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        entries.append((color, label, years[-1], shares[-1]))

    max_year = max(all_years) if all_years else 0
    label_x = max_year + 1.5
    ax.set_xlim(right=label_x + 4)

    entries.sort(key=lambda e: e[3], reverse=True)
    y_min, y_max = ax.get_ylim()
    n = len(entries)
    for i, (color, label, last_year, last_share) in enumerate(entries):
        slot_y = y_max - (i + 0.5) * (y_max - y_min) / max(n, 1)
        ax.annotate(
            label,
            xy=(last_year, last_share),
            xytext=(label_x, slot_y),
            color=color,
            fontsize=8,
            fontweight="normal" if color == _GREY else "bold",
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.6, "alpha": 0.5},
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Share of papers")
    ax.set_title("Medical-imaging AI research: topic share over time")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
