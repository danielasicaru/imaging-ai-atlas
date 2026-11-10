from datetime import date
from pathlib import Path

from imaging_ai_atlas.data.pubmed import PubmedSource

_SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
<MedlineCitation>
<PMID>12345678</PMID>
<Article>
<ArticleTitle>Deep learning for cervical spine fracture detection</ArticleTitle>
<Abstract>
<AbstractText>We propose a model for cervical spine fracture detection on CT.</AbstractText>
</Abstract>
<Journal><JournalIssue><PubDate><Year>2021</Year><Month>Mar</Month><Day>5</Day></PubDate></JournalIssue></Journal>
</Article>
<MeshHeadingList>
<MeshHeading><DescriptorName>Cervical Vertebrae</DescriptorName></MeshHeading>
<MeshHeading><DescriptorName>Deep Learning</DescriptorName></MeshHeading>
</MeshHeadingList>
</MedlineCitation>
</PubmedArticle>
<PubmedArticle>
<MedlineCitation>
<PMID>87654321</PMID>
<Article>
<ArticleTitle>A retraction notice</ArticleTitle>
<Journal><JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue></Journal>
</Article>
<MeshHeadingList>
<MeshHeading><DescriptorName>Cervical Vertebrae</DescriptorName></MeshHeading>
</MeshHeadingList>
</MedlineCitation>
</PubmedArticle>
</PubmedArticleSet>
"""


def test_iter_records_parses_a_well_formed_article(tmp_path: Path) -> None:
    raw_path = tmp_path / "pubmed.xml"
    raw_path.write_text(_SAMPLE_XML, encoding="utf-8")

    records = list(PubmedSource().iter_records(raw_path))

    assert len(records) == 1
    record = records[0]
    assert record.source == "pubmed"
    assert record.external_id == "12345678"
    assert record.title == "Deep learning for cervical spine fracture detection"
    assert "cervical spine fracture detection" in record.abstract
    assert record.categories == ["Cervical Vertebrae", "Deep Learning"]
    assert record.created == date(2021, 3, 5)
    assert record.update_date is None


def test_iter_records_skips_articles_with_no_abstract_and_counts_the_skip(tmp_path: Path) -> None:
    raw_path = tmp_path / "pubmed.xml"
    raw_path.write_text(_SAMPLE_XML, encoding="utf-8")

    source = PubmedSource()
    records = list(source.iter_records(raw_path))

    assert len(records) == 1
    assert source.skipped == 1
