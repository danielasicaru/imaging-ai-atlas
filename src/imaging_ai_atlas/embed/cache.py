import hashlib
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def cache_key(
    model_name: str,
    revision: str | None,
    task_prefix: str,
    max_len: int,
    data_hash: str,
    dims: int | None,
    cleaning_version: int,
) -> str:
    """A short, stable identifier for one (model, config, data, preprocessing) embedding
    run. Hashed rather than used as a literal filename so none of the inputs need
    filesystem-safe characters (model names contain "/"). `cleaning_version` must be part
    of this, not just `data_hash`: a text-cleaning logic change (e.g. a strip_latex regex
    fix) changes what gets embedded without changing the corpus's ids/content, and a cache
    keyed only on data would silently keep serving embeddings cleaned under the old logic.
    """
    raw = f"{model_name}|{revision}|{task_prefix}|{max_len}|{data_hash}|{dims}|{cleaning_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _path(key: str, cache_dir: Path) -> Path:
    return cache_dir / f"{key}.safetensors"


def load_cached(key: str, cache_dir: Path) -> np.ndarray | None:
    path = _path(key, cache_dir)
    if not path.exists():
        return None
    return load_file(str(path))["embeddings"]


def save_cached(key: str, embeddings: np.ndarray, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_file({"embeddings": embeddings}, str(_path(key, cache_dir)))
