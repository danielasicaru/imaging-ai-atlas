import hashlib
from pathlib import Path

import mlflow
import pandas as pd

from imaging_ai_atlas.data.clean_text import CLEANING_VERSION, strip_latex
from imaging_ai_atlas.embed.cache import cache_key, load_cached, save_cached
from imaging_ai_atlas.embed.config import EmbedModelConfig
from imaging_ai_atlas.embed.embedder import EmbeddingModel, embed_texts, truncate_and_normalize
from imaging_ai_atlas.embed.model_loader import SentenceTransformerModel


def corpus_hash(df: pd.DataFrame) -> str:
    """Hash of the papers being embedded - id *and* the text content that actually gets
    embedded, not just the id set. A re-scrape that fixes a truncated abstract for the same
    external_id must invalidate the cache; hashing ids alone would miss that.
    """
    rows = sorted(zip(df["external_id"].astype(str), df["title"], df["abstract"], strict=True))
    raw = "|".join(f"{i}\x1f{t}\x1f{a}" for i, t, a in rows)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def run_embedding_sweep(
    df: pd.DataFrame,
    config: EmbedModelConfig,
    cache_dir: Path,
    model: EmbeddingModel | None = None,
) -> None:
    """Embed `df` with one model config across all its `dims`, caching each result and
    logging one MLflow run per (model, dims) combination. `df` must already be the
    filtered, deduplicated corpus to embed - this function doesn't apply the M2 filter.

    The model only ever runs once, at full dimension: a Matryoshka model's lower-dim
    variants are truncations of that one embedding, not separate encode passes - encoding
    5 times for a 5-way dims ablation would be pure waste. `model` is injectable so this
    orchestration logic (one encode, N cache entries) is testable without a real model.
    """
    texts = [
        strip_latex(f"{title} {abstract}")
        for title, abstract in zip(df["title"], df["abstract"], strict=True)
    ]
    data_hash = corpus_hash(df)
    full_key = cache_key(
        config.name,
        config.revision,
        config.task_prefix,
        config.max_len,
        data_hash,
        None,
        CLEANING_VERSION,
    )
    full_embeddings = load_cached(full_key, cache_dir)
    full_cache_hit = full_embeddings is not None
    if full_embeddings is None:
        active_model = model
        if active_model is None:
            active_model = SentenceTransformerModel(
                config.name,
                revision=config.revision,
                trust_remote_code=config.trust_remote_code,
                max_len=config.max_len,
            )
        full_embeddings = embed_texts(
            active_model, texts, config.task_prefix, config.batch_size, None
        )
        save_cached(full_key, full_embeddings, cache_dir)

    # dims=None reuses the full embedding directly rather than its own load_cached call:
    # it's the same artifact already in hand, not a second cache entry to look up.
    for dims in config.dims:
        if dims is None:
            embeddings, cache_hit = full_embeddings, full_cache_hit
        else:
            key = cache_key(
                config.name,
                config.revision,
                config.task_prefix,
                config.max_len,
                data_hash,
                dims,
                CLEANING_VERSION,
            )
            embeddings = load_cached(key, cache_dir)
            cache_hit = embeddings is not None
            if embeddings is None:
                embeddings = truncate_and_normalize(full_embeddings, dims)
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
