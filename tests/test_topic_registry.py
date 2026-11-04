from pathlib import Path

import pytest

from imaging_ai_atlas.topics.registry import register_topic_model


class FakeTopicModel:
    def save(self, path: Path, **_kwargs: object) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.safetensors").write_text("fake")


def test_register_topic_model_logs_artifacts_and_tags(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    register_topic_model(
        FakeTopicModel(),  # type: ignore[arg-type]
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
