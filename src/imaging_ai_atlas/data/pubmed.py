from collections.abc import Iterator
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

from imaging_ai_atlas.data.schema import PaperRecord

_MONTH_ABBREVIATIONS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


class PubmedSource:
    """Reads a local PubMed `PubmedArticleSet` XML snapshot (NCBI efetch output).

    Articles with no abstract or no MeSH headings are skipped rather than raising -
    `PaperRecord.categories`/`abstract` both require content, but a bulk XML dump can
    contain a handful of stub records (e.g. errata, retraction notices) that lack both;
    one malformed record shouldn't abort ingesting the rest. The skip count is returned
    alongside the records so callers can see how many were dropped.
    """

    def __init__(self) -> None:
        self.skipped = 0

    def iter_records(self, raw_path: Path) -> Iterator[PaperRecord]:
        self.skipped = 0
        tree = ElementTree.parse(raw_path)  # noqa: S314 - trusted local snapshot, not user input
        for article in tree.getroot().iterfind(".//PubmedArticle"):
            record = self._to_record(article)
            if record is None:
                self.skipped += 1
                continue
            yield record

    @staticmethod
    def _to_record(article: ElementTree.Element) -> PaperRecord | None:
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            return None

        title_el = article.find(".//ArticleTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""

        abstract_parts = [(node.text or "").strip() for node in article.iterfind(".//AbstractText")]
        abstract = " ".join(part for part in abstract_parts if part)
        if not abstract:
            return None

        categories = [
            descriptor.text.strip()
            for descriptor in article.iterfind(".//MeshHeading/DescriptorName")
            if descriptor.text
        ]
        if not categories:
            return None

        created = PubmedSource._parse_pub_date(article)
        if created is None:
            return None

        return PaperRecord(
            source="pubmed",
            external_id=pmid_el.text.strip(),
            title=title,
            abstract=abstract,
            categories=categories,
            created=created,
            update_date=None,
        )

    @staticmethod
    def _parse_pub_date(article: ElementTree.Element) -> date | None:
        pub_date = article.find(".//Journal/JournalIssue/PubDate")
        if pub_date is None:
            return None

        year_el = pub_date.find("Year")
        if year_el is None or not year_el.text:
            return None
        year = int(year_el.text)

        month_el = pub_date.find("Month")
        month = 1
        if month_el is not None and month_el.text:
            month = _MONTH_ABBREVIATIONS.get(month_el.text.strip()) or (
                int(month_el.text) if month_el.text.strip().isdigit() else 1
            )

        day_el = pub_date.find("Day")
        day = int(day_el.text) if day_el is not None and day_el.text else 1

        return date(year, month, day)
