from pathlib import Path

import yaml
from pydantic import BaseModel


class UmapConfig(BaseModel):
    # n_components=5 is for clustering, not the 2D display map - that's a separate UMAP
    # call (story/map2d.py), tuned for human-readable layout rather than HDBSCAN's
    # density assumptions.
    n_components: int = 5
    n_neighbors: int = 15
    min_dist: float = 0.0
    metric: str = "cosine"


class HdbscanConfig(BaseModel):
    min_cluster_size: int = 15
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"


class VectorizerConfig(BaseModel):
    stop_words: str = "english"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 5


class TopicModelConfig(BaseModel):
    umap: UmapConfig = UmapConfig()
    hdbscan: HdbscanConfig = HdbscanConfig()
    vectorizer: VectorizerConfig = VectorizerConfig()
    seeds: list[int] = [0, 1, 2, 3, 4]
    outlier_reduction_strategy: str = "c-tf-idf"
    top_n_words: int = 10


class EmbeddingVariant(BaseModel):
    """One (model, dims) combination already produced by `atlas embed` - the cache key
    fields needed to look it up, not the embedding data itself.
    """

    model_name: str
    revision: str | None = None
    task_prefix: str = ""
    max_len: int
    dims: int | None = None


class TopicSweepConfig(BaseModel):
    topic_model: TopicModelConfig = TopicModelConfig()
    embedding_variants: list[EmbeddingVariant]


def load_topic_sweep_config(path: Path) -> TopicSweepConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TopicSweepConfig.model_validate(raw)
