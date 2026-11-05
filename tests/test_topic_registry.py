from pathlib import Path

import pytest

from imaging_ai_atlas.topics.registry import register_topic_model


class FakeTopicModel:
    def __init__(self) -> None:
        self.save_kwargs: dict[str, object] = {}

    def save(self, path: Path, **kwargs: object) -> None:
        self.save_kwargs = kwargs
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.safetensors").write_text("fake")


def test_register_topic_model_logs_artifacts_and_tags(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    fake_model = FakeTopicModel()

    register_topic_model(
        fake_model,  # type: ignore[arg-type]
        tmp_path / "model_dir",
        data_hash="deadbeef",
        embed_model="nomic-ai/nomic-embed-text-v1.5",
        git_sha="abc123",
    )

    runs = mlflow.search_runs(search_all_experiments=True)
    assert runs.iloc[0]["tags.data_hash"] == "deadbeef"
    assert runs.iloc[0]["tags.embed_model"] == "nomic-ai/nomic-embed-text-v1.5"
    assert runs.iloc[0]["tags.git_sha"] == "abc123"
    assert (tmp_path / "model_dir" / "model.safetensors").exists()
    # The embedding model must not be bundled - inference on new documents has to go
    # through M3's pinned loader (same revision/task_prefix), not whatever BERTopic would
    # try to download from a bare model-name pointer.
    assert fake_model.save_kwargs["save_embedding_model"] is False
