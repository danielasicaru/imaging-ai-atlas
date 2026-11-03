import mlflow
import numpy as np
import pandas as pd

from imaging_ai_atlas.embed.config import EmbedModelConfig
from imaging_ai_atlas.embed.pipeline import corpus_hash, run_embedding_sweep


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.calls += 1
        return np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (len(texts), 1))


def _tiny_corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "external_id": ["1", "2"],
            "title": ["t1", "t2"],
            "abstract": [r"a1 with \emph{markup}", "a2"],
        }
    )


def test_corpus_hash_is_order_independent() -> None:
    df_a = pd.DataFrame({"external_id": ["1", "2", "3"]})
    df_b = pd.DataFrame({"external_id": ["3", "1", "2"]})

    assert corpus_hash(df_a) == corpus_hash(df_b)


def test_corpus_hash_changes_when_the_id_set_changes() -> None:
    df_a = pd.DataFrame({"external_id": ["1", "2", "3"]})
    df_b = pd.DataFrame({"external_id": ["1", "2", "4"]})

    assert corpus_hash(df_a) != corpus_hash(df_b)


def test_run_embedding_sweep_encodes_once_regardless_of_dims_count(tmp_path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    config = EmbedModelConfig(name="fake-model", max_len=16, dims=[None, 2])
    model = _FakeModel()

    run_embedding_sweep(_tiny_corpus(), config, tmp_path / "cache", model=model)

    assert model.calls == 1


def test_run_embedding_sweep_is_idempotent_via_the_cache(tmp_path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    config = EmbedModelConfig(name="fake-model", max_len=16, dims=[None, 2])
    cache_dir = tmp_path / "cache"
    df = _tiny_corpus()

    run_embedding_sweep(df, config, cache_dir, model=_FakeModel())
    second_model = _FakeModel()
    run_embedding_sweep(df, config, cache_dir, model=second_model)

    assert second_model.calls == 0
