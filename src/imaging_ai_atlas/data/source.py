from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from imaging_ai_atlas.data.schema import PaperRecord


class Source(Protocol):
    """Anything that can turn a raw snapshot into validated PaperRecords.

    `ingest` depends only on this, never on a concrete source — adding a new source
    (e.g. PubMed) means writing a class that satisfies this shape, not touching ingest.
    """

    def iter_records(self, raw_path: Path) -> Iterator[PaperRecord]: ...
