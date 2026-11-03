from pathlib import Path

import yaml
from pydantic import BaseModel


class EmbedModelConfig(BaseModel):
    name: str
    revision: str | None = None
    trust_remote_code: bool = False
    task_prefix: str = ""
    max_len: int
    batch_size: int = 64
    dims: list[int | None] = [None]


class EmbedConfig(BaseModel):
    models: list[EmbedModelConfig]


def load_embed_config(path: Path) -> EmbedConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EmbedConfig.model_validate(raw)
