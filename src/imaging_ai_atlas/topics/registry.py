from pathlib import Path

import mlflow
from bertopic import BERTopic


def register_topic_model(
    topic_model: BERTopic, save_dir: Path, data_hash: str, embed_model: str, git_sha: str
) -> None:
    """Save the chosen topic model as a standard MLflow artifact (safetensors, with
    c-TF-IDF) - not "models from code", which MLflow's own docs reserve for weightless
    logic like GenAI agents, not trained models like this one. The embedding model itself
    is not bundled (`save_embedding_model=False`): reproducing inference on new documents
    must go through M3's pinned loader (same revision/task_prefix), not a bare model-name
    string BERTopic would try to download itself.
    """
    topic_model.save(
        save_dir, serialization="safetensors", save_ctfidf=True, save_embedding_model=False
    )
    with mlflow.start_run(run_name="topic-model-registration"):
        mlflow.set_tags({"data_hash": data_hash, "embed_model": embed_model, "git_sha": git_sha})
        mlflow.log_artifacts(str(save_dir), artifact_path="topic_model")
