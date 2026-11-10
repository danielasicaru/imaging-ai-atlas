from pathlib import Path

import pandas as pd


def combine_corpora(paths: list[Path], out_path: Path) -> int:
    """Concatenate multiple `PaperRecord`-shaped parquet files (e.g. one per source) into
    one corpus, deduping on `(source, external_id)` - the same key `ingest()` uses, since
    a paper could in principle appear in more than one snapshot of the same source.
    """
    frames = [pd.read_parquet(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["source", "external_id"], keep="first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    return len(combined)
