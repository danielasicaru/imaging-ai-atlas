import numpy as np

from imaging_ai_atlas.topics.config import (
    HdbscanConfig,
    TopicModelConfig,
    UmapConfig,
    VectorizerConfig,
)
from imaging_ai_atlas.topics.model import fit_topic_model, topic_keywords


def _tiny_corpus() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.05, size=(15, 8))
    cluster_b = rng.normal(loc=5.0, scale=0.05, size=(15, 8))
    embeddings = np.vstack([cluster_a, cluster_b]).astype(np.float32)
    texts = ["mri brain scan tumor segmentation deep learning"] * 15 + [
        "ct lung nodule detection radiograph model"
    ] * 15
    return embeddings, texts


def _tiny_config() -> TopicModelConfig:
    return TopicModelConfig(
        umap=UmapConfig(n_components=2, n_neighbors=5, min_dist=0.0, metric="euclidean"),
        hdbscan=HdbscanConfig(min_cluster_size=5),
        vectorizer=VectorizerConfig(min_df=1),
        seeds=[0],
    )


def test_fit_topic_model_separates_two_distinct_clusters() -> None:
    embeddings, texts = _tiny_corpus()

    _topic_model, topics = fit_topic_model(embeddings, texts, seed=0, config=_tiny_config())

    assert len(topics) == 30
    non_outlier_topics = {t for t in topics if t != -1}
    assert len(non_outlier_topics) >= 2


def test_topic_keywords_excludes_the_outlier_bucket() -> None:
    embeddings, texts = _tiny_corpus()
    topic_model, topics = fit_topic_model(embeddings, texts, seed=0, config=_tiny_config())

    keywords = topic_keywords(topic_model)

    non_outlier_topic_count = len({t for t in topics if t != -1})
    assert len(keywords) == non_outlier_topic_count
    assert all(len(topic_words) > 0 for topic_words in keywords)
