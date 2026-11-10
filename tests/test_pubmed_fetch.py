from pathlib import Path
from unittest.mock import MagicMock, patch

from imaging_ai_atlas.data.pubmed_fetch import fetch_pubmed_xml


def _mock_response(json_data=None, text_data=None):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    if json_data is not None:
        response.json.return_value = json_data
    if text_data is not None:
        response.text = text_data
    return response


@patch("imaging_ai_atlas.data.pubmed_fetch.time.sleep")
@patch("imaging_ai_atlas.data.pubmed_fetch.requests.get")
def test_fetch_pubmed_xml_writes_combined_article_set(mock_get, mock_sleep, tmp_path: Path) -> None:
    esearch_response = _mock_response(json_data={"esearchresult": {"idlist": ["1", "2"]}})
    efetch_response = _mock_response(
        text_data='<?xml version="1.0"?><PubmedArticleSet>'
        "<PubmedArticle><MedlineCitation><PMID>1</PMID></MedlineCitation></PubmedArticle>"
        "<PubmedArticle><MedlineCitation><PMID>2</PMID></MedlineCitation></PubmedArticle>"
        "</PubmedArticleSet>"
    )
    mock_get.side_effect = [esearch_response, efetch_response]

    out_path = tmp_path / "pubmed.xml"
    count = fetch_pubmed_xml("cervical spine fracture", out_path, email="me@example.com")

    assert count == 2
    content = out_path.read_text(encoding="utf-8")
    assert content.count("<PubmedArticle>") == 2
    assert content.startswith('<?xml version="1.0"?>')
    assert content.strip().endswith("</PubmedArticleSet>")
    assert mock_get.call_count == 2
    assert mock_sleep.call_count == 2


@patch("imaging_ai_atlas.data.pubmed_fetch.time.sleep")
@patch("imaging_ai_atlas.data.pubmed_fetch.requests.get")
def test_fetch_pubmed_xml_batches_efetch_requests(mock_get, mock_sleep, tmp_path: Path) -> None:
    pmids = [str(i) for i in range(250)]
    esearch_response = _mock_response(json_data={"esearchresult": {"idlist": pmids}})
    efetch_response = _mock_response(
        text_data="<PubmedArticleSet><PubmedArticle></PubmedArticle></PubmedArticleSet>"
    )
    mock_get.side_effect = [esearch_response, efetch_response, efetch_response]

    out_path = tmp_path / "pubmed.xml"
    count = fetch_pubmed_xml("x", out_path, email="me@example.com", max_results=250)

    assert count == 250
    # 1 esearch call + 2 efetch batches (200 + 50)
    assert mock_get.call_count == 3
