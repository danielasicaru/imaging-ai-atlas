import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from imaging_ai_atlas.data.clean_text import CLEANING_VERSION, strip_latex
from imaging_ai_atlas.embed.cache import cache_key, load_cached
from imaging_ai_atlas.topics.config import EmbeddingVariant, TopicModelConfig
from imaging_ai_atlas.topics.metrics import npmi_coherence, outlier_rate, topic_diversity
from imaging_ai_atlas.topics.model import fit_topic_model, topic_keywords
from imaging_ai_atlas.topics.stability import mean_pairwise_ari


def load_variant_embeddings(
    variant: EmbeddingVariant, data_hash: str, cache_dir: Path
) -> np.ndarray:
    """Look up an embedding already produced by `atlas embed` (M3), using the same cache
    key scheme - this module never computes embeddings itself.
    """
    key = cache_key(
        variant.model_name,
        variant.revision,
        variant.task_prefix,
        variant.max_len,
        data_hash,
        variant.dims,
        CLEANING_VERSION,
    )
    embeddings = load_cached(key, cache_dir)
    if embeddings is None:
        raise FileNotFoundError(
            f"No cached embedding for {variant.model_name} (dims={variant.dims}) - "
            "run `atlas embed` first."
        )
    return embeddings


def evaluate_variant(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    variant: EmbeddingVariant,
    config: TopicModelConfig,
) -> None:
    """Fit once at config.seeds[0] for diversity/NPMI/outlier-rate/runtime, then refit at
    every remaining seed to measure stability (mean pairwise ARI). Logs one MLflow run per
    variant with every metric this milestone's plan requires.
    """
    texts = [
        strip_latex(f"{title} {abstract}")
        for title, abstract in zip(df["title"], df["abstract"], strict=True)
    ]

    start = time.perf_counter()
    topic_model, topics = fit_topic_model(embeddings, texts, config.seeds[0], config)
    runtime_s = time.perf_counter() - start

    keywords = topic_keywords(topic_model)
    analyze = topic_model.vectorizer_model.build_analyzer()
    tokenized_docs = [analyze(text) for text in texts]

    diversity = topic_diversity(keywords)
    npmi = npmi_coherence(keywords, tokenized_docs) if keywords else float("nan")
    outlier_rate_before = outlier_rate(topics)
    # reduce_outliers raises ValueError on a corpus with zero outliers rather than being a
    # no-op - nothing to reduce means the "after" rate is trivially the same as "before".
    if outlier_rate_before > 0:
        reduced_topics = topic_model.reduce_outliers(
            texts, topics, strategy=config.outlier_reduction_strategy
        )
        outlier_rate_after = outlier_rate(reduced_topics)
    else:
        outlier_rate_after = outlier_rate_before

    all_assignments = [topics]
    for seed in config.seeds[1:]:
        _, seed_topics = fit_topic_model(embeddings, texts, seed, config)
        all_assignments.append(seed_topics)
    ari_mean = mean_pairwise_ari(all_assignments) if len(all_assignments) > 1 else None

    with mlflow.start_run(run_name=f"{variant.model_name}-{variant.dims}"):
        mlflow.log_params(
            {
                "model_name": variant.model_name,
                "dims": variant.dims,
                "n_topics": len(keywords),
                "n_docs": len(df),
                "seeds": config.seeds,
            }
        )
        metrics = {
            "diversity": diversity,
            "npmi": npmi,
            "outlier_rate_before": outlier_rate_before,
            "outlier_rate_after": outlier_rate_after,
            "runtime_s": runtime_s,
        }
        if ari_mean is not None:
            metrics["ari_mean"] = ari_mean
        mlflow.log_metrics(metrics)
