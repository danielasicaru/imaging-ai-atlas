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


def _ids_df(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"external_id": ids, "title": [f"t{i}" for i in ids], "abstract": [f"a{i}" for i in ids]}
    )


def test_corpus_hash_is_order_independent() -> None:
    df_a = _ids_df(["1", "2", "3"])
    df_b = _ids_df(["3", "1", "2"])

    assert corpus_hash(df_a) == corpus_hash(df_b)


def test_corpus_hash_changes_when_the_id_set_changes() -> None:
    df_a = _ids_df(["1", "2", "3"])
    df_b = _ids_df(["1", "2", "4"])

    assert corpus_hash(df_a) != corpus_hash(df_b)


def test_corpus_hash_changes_when_text_content_changes_for_the_same_ids() -> None:
    # A re-scrape that fixes a truncated abstract must invalidate the cache, even though
    # the set of external_ids is unchanged.
    df_a = _ids_df(["1", "2"])
    df_b = df_a.copy()
    df_b.loc[0, "abstract"] = "a corrected, longer abstract"

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


def test_run_embedding_sweep_constructs_the_real_model_with_config_params(
    tmp_path, monkeypatch
) -> None:
    # Closes the gap where model=None (the production path) was only ever exercised with
    # real network calls, never asserted on: verifies EmbedModelConfig's revision/
    # trust_remote_code/max_len actually reach SentenceTransformer's constructor and
    # max_seq_length, without downloading anything.
    captured: dict[str, object] = {}

    class _FakeSentenceTransformer:
        def __init__(self, name: str, revision: str | None, trust_remote_code: bool) -> None:
            captured["name"] = name
            captured["revision"] = revision
            captured["trust_remote_code"] = trust_remote_code
            self.max_seq_length = 0
            instances.append(self)

        def encode(self, texts: list[str], batch_size: int, convert_to_numpy: bool) -> np.ndarray:
            return np.tile(np.array([1.0, 2.0]), (len(texts), 1))

    instances: list[_FakeSentenceTransformer] = []

    monkeypatch.setattr(
        "imaging_ai_atlas.embed.model_loader.SentenceTransformer", _FakeSentenceTransformer
    )
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    config = EmbedModelConfig(
        name="nomic-ai/nomic-embed-text-v1.5",
        revision="abc123",
        trust_remote_code=True,
        max_len=99,
        dims=[None],
    )

    run_embedding_sweep(_tiny_corpus(), config, tmp_path / "cache")

    assert captured["name"] == "nomic-ai/nomic-embed-text-v1.5"
    assert captured["revision"] == "abc123"
    assert captured["trust_remote_code"] is True
    assert instances[0].max_seq_length == 99
