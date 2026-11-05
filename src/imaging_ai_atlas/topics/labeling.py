from collections.abc import Mapping
from typing import Protocol

import numpy as np
import pandas as pd


class TopicKeywordLookup(Protocol):
    """Whatever this module actually needs from a fitted topic model: per-topic
    keywords. Narrower than the full BERTopic API, same DI pattern as `EmbeddingModel`
    and `DimensionalityReducer` elsewhere in this project, so tests don't need a real fit.
    Return type matches BERTopic's actual runtime behavior, not its own (wrong) type hint
    for `get_topic` - same caveat as `topic_keywords` in `topics/model.py`.
    """

    def get_topic(
        self, topic_id: int, /
    ) -> Mapping[str, tuple[str, float]] | list[tuple[str, float]] | bool: ...


def build_topic_label_sample(
    topic_model: TopicKeywordLookup, topics: list[int], texts: list[str], n_topics: int, seed: int
) -> pd.DataFrame:
    """The `n_topics` largest real topics (an incoherent large topic damages the overall
    story more than an incoherent small one) with top keywords and example documents, for
    a human coherence spot-check - same spirit as the filter stage's labeling CSV, but for
    topics rather than filter decisions. The outlier bucket (-1) is excluded, it isn't a
    topic. Ties in size are broken by ascending topic_id, so the selection is fully
    deterministic rather than relying on `value_counts`'s incidental tie order.
    """
    sizes: dict[int, int] = pd.Series(topics).value_counts().to_dict()
    top_ids = sorted((t for t in sizes if t != -1), key=lambda t: (-sizes[t], t))[:n_topics]

    rng = np.random.default_rng(seed)
    rows = []
    for topic_id in top_ids:
        doc_idxs = [i for i, t in enumerate(topics) if t == topic_id]
        chosen = rng.choice(doc_idxs, size=min(3, len(doc_idxs)), replace=False)
        examples = [texts[i][:200] for i in chosen]
        # `topic_id` always comes from `topics` itself, so `get_topic` should always know
        # about it; if a model ever returned its `False`-for-unknown-topic sentinel here
        # anyway (e.g. drift between fit state and the `topics` labels), this degrades to
        # an empty keyword string rather than raising - a deliberate, permissive choice to
        # keep the sample generator usable even on partially-inconsistent input, not an
        # oversight.
        words = topic_model.get_topic(topic_id)
        # BERTopic returns either a list of (word, score) tuples or a word->score mapping
        # depending on version - normalize both to the word strings, in order.
        if isinstance(words, list):
            keywords = ", ".join(word for word, _score in words)
        elif isinstance(words, Mapping):
            keywords = ", ".join(words)
        else:
            keywords = ""
        rows.append(
            {
                "topic_id": topic_id,
                "size": sizes[topic_id],
                "keywords": keywords,
                "example_1": examples[0] if len(examples) > 0 else "",
                "example_2": examples[1] if len(examples) > 1 else "",
                "example_3": examples[2] if len(examples) > 2 else "",
                "coherent": "",
                "suggested_label": "",
            }
        )
    return pd.DataFrame(rows)
