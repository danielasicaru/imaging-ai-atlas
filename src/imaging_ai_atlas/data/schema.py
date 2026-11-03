from datetime import date

from pydantic import BaseModel, Field


class PaperRecord(BaseModel):
    """One paper, normalized to the shape every pipeline stage after `ingest` can trust.

    Any `Source` must produce these, regardless of what its raw format looks like.
    """

    source: str
    external_id: str
    title: str
    abstract: str
    categories: list[str] = Field(min_length=1)
    created: date
    update_date: date | None = None
