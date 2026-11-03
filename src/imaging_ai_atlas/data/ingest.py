from pathlib import Path

import pandas as pd

from imaging_ai_atlas.data.source import Source


def ingest(source: Source, raw_path: Path, out_path: Path) -> int:
    """Stream records from `source` straight to Parquet. Returns the row count written."""
    records = [record.model_dump() for record in source.iter_records(raw_path)]
    df = pd.DataFrame.from_records(records)
    # Snapshot sources are expected to carry one row per (source, external_id) already (see
    # ADR 0001) — this is a cheap assertion of that assumption, not a substitute for it.
    df = df.drop_duplicates(subset=["source", "external_id"], keep="first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)
