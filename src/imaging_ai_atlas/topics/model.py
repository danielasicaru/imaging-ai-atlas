from typing import Protocol

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from imaging_ai_atlas.topics.config import TopicModelConfig


class DimensionalityReducer(Protocol):
    """Whatever BERTopic's `umap_model` slot actually needs: fit_transform. UMAP and PCA
    both satisfy this, which is what makes the PCA-5D control experiment a drop-in swap
    rather than a separate pipeline.
    """

    def fit_transform(self, embeddings: np.ndarray, /) -> np.ndarray: ...


def _build_topic_model(reducer: DimensionalityReducer, config: TopicModelConfig) -> BERTopic:
    """Construct a BERTopic instance with every component explicit - no hidden defaults,
    per the project's reproducibility rule. HDBSCAN and CountVectorizer are deterministic
    given their inputs; only the reducer (UMAP or PCA) takes a seed.
    """
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.hdbscan.min_cluster_size,
        metric=config.hdbscan.metric,
        cluster_selection_method=config.hdbscan.cluster_selection_method,
    )
    vectorizer_model = CountVectorizer(
        stop_words=config.vectorizer.stop_words,
        ngram_range=config.vectorizer.ngram_range,
        min_df=config.vectorizer.min_df,
    )
    return BERTopic(
        umap_model=reducer,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=config.top_n_words,
        calculate_probabilities=False,
    )


def fit_topic_model(
    embeddings: np.ndarray, texts: list[str], seed: int, config: TopicModelConfig
) -> tuple[BERTopic, list[int]]:
    """Fit a topic model on pre-computed embeddings (from the M3 cache) and raw texts
    (for c-TF-IDF/keyword extraction - a separate concern from what was embedded).
    Fixing UMAP's `random_state` disables its parallelism, a documented tradeoff for
    reproducibility, not an oversight.
    """
    reducer = UMAP(
        n_components=config.umap.n_components,
        n_neighbors=config.umap.n_neighbors,
        min_dist=config.umap.min_dist,
        metric=config.umap.metric,
        random_state=seed,
    )
    topic_model = _build_topic_model(reducer, config)
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    return topic_model, topics


def fit_topic_model_with_pca(
    embeddings: np.ndarray, texts: list[str], seed: int, config: TopicModelConfig
) -> tuple[BERTopic, list[int]]:
    """Control experiment: linear PCA instead of UMAP for the same n_components, HDBSCAN
    and vectorizer unchanged. Quantifies whether UMAP's non-linear structure actually
    earns its place over a much simpler, deterministic linear projection.
    """
    reducer = PCA(n_components=config.umap.n_components, random_state=seed)
    topic_model = _build_topic_model(reducer, config)
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    return topic_model, topics


def topic_keywords(topic_model: BERTopic) -> list[list[str]]:
    """Top keywords per real topic, excluding -1 (the outlier bucket, not a topic)."""
    # bertopic's own type hints for get_topics() are wrong (claims Mapping[str, tuple]);
    # its actual runtime return is dict[int, list[tuple[str, float]]], per its own docs.
    topics = topic_model.get_topics()
    return [
        [word for word, _score in words]  # pyright: ignore[reportGeneralTypeIssues]
        for topic_id, words in topics.items()
        if topic_id != -1
    ]
