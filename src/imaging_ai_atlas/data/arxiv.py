import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from imaging_ai_atlas.data.schema import PaperRecord

# arXiv's version timestamp format, e.g. "Mon, 2 Apr 2007 19:18:42 GMT".
_VERSION_DATE_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"


class ArxivSource:
    """Reads the Kaggle arXiv metadata snapshot (one JSON object per line)."""

    def iter_records(self, raw_path: Path) -> Iterator[PaperRecord]:
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                yield self._to_record(raw)

    @staticmethod
    def _to_record(raw: dict) -> PaperRecord:
        # Date = v1 `created`, not `update_date`: see ADR 0001. update_date moves every time
        # metadata is edited, which would misdate a paper's first appearance.
        first_version_created = raw["versions"][0]["created"]
        created = datetime.strptime(first_version_created, _VERSION_DATE_FORMAT).date()

        update_date = (
            datetime.strptime(raw["update_date"], "%Y-%m-%d").date()
            if raw.get("update_date")
            else None
        )

        return PaperRecord(
            source="arxiv",
            external_id=raw["id"],
            title=raw["title"].strip(),
            abstract=raw["abstract"].strip(),
            categories=raw["categories"].split(),
            created=created,
            update_date=update_date,
        )
