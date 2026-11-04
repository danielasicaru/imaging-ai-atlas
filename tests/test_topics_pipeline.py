import numpy as np
import pandas as pd
import pytest

from imaging_ai_atlas.data.clean_text import CLEANING_VERSION
from imaging_ai_atlas.embed.cache import cache_key, save_cached
from imaging_ai_atlas.topics.config import (
    EmbeddingVariant,
    HdbscanConfig,
    TopicModelConfig,
    UmapConfig,
    VectorizerConfig,
)
from imaging_ai_atlas.topics.pipeline import evaluate_variant, load_variant_embeddings


def test_load_variant_embeddings_raises_a_clear_error_when_uncached(tmp_path) -> None:
    variant = EmbeddingVariant(model_name="some-model", max_len=256, dims=None)

    with pytest.raises(FileNotFoundError, match="atlas embed"):
        load_variant_embeddings(variant, data_hash="deadbeef", cache_dir=tmp_path)


def test_load_variant_embeddings_round_trips_through_the_m3_cache(tmp_path) -> None:
    variant = EmbeddingVariant(
        model_name="some-model", revision="abc", task_prefix="x: ", max_len=256, dims=128
    )
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    key = cache_key(
        variant.model_name,
        variant.revision,
        variant.task_prefix,
        variant.max_len,
        "deadbeef",
        variant.dims,
        CLEANING_VERSION,
    )
    save_cached(key, embeddings, tmp_path)

    loaded = load_variant_embeddings(variant, data_hash="deadbeef", cache_dir=tmp_path)

    np.testing.assert_array_equal(loaded, embeddings)


def test_evaluate_variant_handles_every_document_being_an_outlier(tmp_path) -> None:
    # min_cluster_size bigger than the corpus guarantees HDBSCAN finds zero real clusters -
    # reduce_outliers has nothing to reduce into (no real topic to reassign an outlier to),
    # which must be skipped rather than attempted, just like the zero-outliers case.
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(30, 8)).astype(np.float32)
    df = pd.DataFrame(
        {"title": ["paper about things"] * 30, "abstract": ["an abstract of some kind"] * 30}
    )
    variant = EmbeddingVariant(model_name="fake-model", max_len=256, dims=None)
    config = TopicModelConfig(
        umap=UmapConfig(n_components=2, n_neighbors=5, min_dist=0.0, metric="euclidean"),
        hdbscan=HdbscanConfig(min_cluster_size=100),
        vectorizer=VectorizerConfig(min_df=1),
        seeds=[0],
    )

    evaluate_variant(df, embeddings, variant, config)

    runs = mlflow.search_runs(search_all_experiments=True)
    assert runs.iloc[0]["metrics.outlier_rate_before"] == 1.0
    assert runs.iloc[0]["metrics.outlier_rate_after"] == 1.0


def test_evaluate_variant_runs_end_to_end_and_logs_mlflow(tmp_path) -> None:
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.05, size=(15, 8))
    cluster_b = rng.normal(loc=5.0, scale=0.05, size=(15, 8))
    embeddings = np.vstack([cluster_a, cluster_b]).astype(np.float32)
    df = pd.DataFrame(
        {
            "title": ["mri brain scan tumor segmentation"] * 15
            + ["ct lung nodule detection radiograph"] * 15,
            "abstract": ["deep learning model evaluation"] * 30,
        }
    )
    variant = EmbeddingVariant(model_name="fake-model", max_len=256, dims=None)
    config = TopicModelConfig(
        umap=UmapConfig(n_components=2, n_neighbors=5, min_dist=0.0, metric="euclidean"),
        hdbscan=HdbscanConfig(min_cluster_size=5),
        vectorizer=VectorizerConfig(min_df=1),
        seeds=[0, 1],
    )

    evaluate_variant(df, embeddings, variant, config)

    runs = mlflow.search_runs(search_all_experiments=True)
    assert len(runs) == 1
    assert runs.iloc[0]["params.model_name"] == "fake-model"
    assert "metrics.diversity" in runs.columns
    assert "metrics.ari_mean" in runs.columns
