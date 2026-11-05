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


def plot_topic_map(
    coords: np.ndarray,
    topics: list[int],
    topic_labels: dict[int, str],
    save_path: Path,
    n_highlighted: int = 10,
) -> None:
    """2D scatter of every document, colored by its topic for the `n_highlighted` largest
    topics (direct-labeled at each topic's centroid) and grey for everything else (smaller
    topics and the outlier bucket) - highlighting all ~190 topics at once would be unreadable
    noise, not a story.
    """
    topics_arr = np.asarray(topics)
    sizes = pd.Series(topics).value_counts()
    highlighted = [t for t in sizes.index if t != -1][:n_highlighted]

    fig, ax = plt.subplots(figsize=(10, 8))
    other_mask = ~np.isin(topics_arr, highlighted)
    ax.scatter(coords[other_mask, 0], coords[other_mask, 1], c=_GREY, s=4, alpha=0.4, linewidths=0)

    for i, topic_id in enumerate(highlighted):
        mask = topics_arr == topic_id
        color = _HIGHLIGHT_CMAP(i % _HIGHLIGHT_CMAP.N)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], s=6, alpha=0.8, linewidths=0)
        centroid_x, centroid_y = coords[mask].mean(axis=0)
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        ax.annotate(
            label,
            (centroid_x, centroid_y),
            fontsize=8,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Medical-imaging AI research: topic map")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_topic_dynamics(
    share: pd.DataFrame, topic_labels: dict[int, str], save_path: Path, n_highlighted: int = 8
) -> None:
    """One line per highlighted topic (its share of the corpus per year), direct-labeled
    at the line's last point instead of a legend; every other topic is summed into a
    single grey "other topics" line so the chart still reflects the full corpus share.
    """
    records: list[dict[str, int | float]] = share[["year", "topic_id", "count", "share"]].to_dict(
        "records"
    )  # type: ignore[assignment]

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

    fig, ax = plt.subplots(figsize=(10, 6))

    other_years = sorted(other_by_year)
    other_shares = [other_by_year[y] for y in other_years]
    ax.plot(other_years, other_shares, color=_GREY, linewidth=1.5)
    if other_years:
        ax.annotate("other topics", (other_years[-1], other_shares[-1]), color=_GREY, fontsize=8)

    for i, topic_id in enumerate(highlighted):
        points = sorted(series_by_topic[topic_id])
        years = [p[0] for p in points]
        shares = [p[1] for p in points]
        color = _HIGHLIGHT_CMAP(i % _HIGHLIGHT_CMAP.N)
        ax.plot(years, shares, color=color, linewidth=2)
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        ax.annotate(label, (years[-1], shares[-1]), color=color, fontsize=8, fontweight="bold")

    ax.set_xlabel("Year")
    ax.set_ylabel("Share of papers")
    ax.set_title("Medical-imaging AI research: topic share over time")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
