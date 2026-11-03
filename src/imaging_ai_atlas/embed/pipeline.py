import hashlib
from pathlib import Path

import mlflow
import pandas as pd

from imaging_ai_atlas.data.clean_text import strip_latex
from imaging_ai_atlas.embed.cache import cache_key, load_cached, save_cached
from imaging_ai_atlas.embed.config import EmbedModelConfig
from imaging_ai_atlas.embed.embedder import embed_texts
from imaging_ai_atlas.embed.model_loader import SentenceTransformerModel


def corpus_hash(df: pd.DataFrame) -> str:
    """Hash of the *set* of papers being embedded, not the whole frame - stable across row
    order and unaffected by columns the embedding step doesn't read.
    """
    ids = sorted(df["external_id"].astype(str))
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]


def run_embedding_sweep(df: pd.DataFrame, config: EmbedModelConfig, cache_dir: Path) -> None:
    """Embed `df` with one model config across all its `dims`, caching each result and
    logging one MLflow run per (model, dims) combination. `df` must already be the
    filtered, deduplicated corpus to embed - this function doesn't apply the M2 filter.
    """
    texts = [
        strip_latex(f"{title} {abstract}")
        for title, abstract in zip(df["title"], df["abstract"], strict=True)
    ]
    data_hash = corpus_hash(df)
    model = SentenceTransformerModel(
        config.name, revision=config.revision, trust_remote_code=config.trust_remote_code
    )

    for dims in config.dims:
        key = cache_key(
            config.name, config.revision, config.task_prefix, config.max_len, data_hash, dims
        )
        embeddings = load_cached(key, cache_dir)
        cache_hit = embeddings is not None
        if embeddings is None:
            embeddings = embed_texts(model, texts, config.task_prefix, config.batch_size, dims)
            save_cached(key, embeddings, cache_dir)

        with mlflow.start_run(run_name=f"{config.name}-{dims}"):
            mlflow.log_params(
                {
                    "model_name": config.name,
                    "revision": config.revision,
                    "task_prefix": config.task_prefix,
                    "max_len": config.max_len,
                    "batch_size": config.batch_size,
                    "dims": dims,
                    "data_hash": data_hash,
                    "cache_hit": cache_hit,
                    "n_papers": len(df),
                }
            )
            mlflow.log_metric("embedding_dim", embeddings.shape[1])
