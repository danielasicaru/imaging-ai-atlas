import time
from pathlib import Path

import requests

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_EFETCH_BATCH_SIZE = 200
# NCBI allows <=3 requests/second without an API key - sleeping between every request
# (not just every batch) keeps this safe regardless of how many batches efetch needs.
_MIN_SECONDS_BETWEEN_REQUESTS = 0.34


def fetch_pubmed_xml(
    query: str,
    raw_path: Path,
    email: str,
    max_results: int = 2000,
    api_key: str | None = None,
) -> int:
    """Search PubMed for `query` via NCBI E-utilities and write the raw `PubmedArticleSet`
    XML for every matching PMID to `raw_path`. Returns the number of PMIDs fetched.

    `email` is required by NCBI's E-utilities usage policy, not an authentication
    mechanism - it identifies the requester in case NCBI needs to contact them about
    usage patterns.
    """
    pmids = _esearch(query, email, max_results, api_key)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n<PubmedArticleSet>\n')
        for batch_start in range(0, len(pmids), _EFETCH_BATCH_SIZE):
            batch = pmids[batch_start : batch_start + _EFETCH_BATCH_SIZE]
            xml_fragment = _efetch(batch, email, api_key)
            f.write(xml_fragment)
            f.write("\n")
        f.write("</PubmedArticleSet>\n")

    return len(pmids)


def _esearch(query: str, email: str, max_results: int, api_key: str | None) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    response = requests.get(f"{_EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
    response.raise_for_status()
    time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS)
    return list(response.json()["esearchresult"]["idlist"])


def _efetch(pmids: list[str], email: str, api_key: str | None) -> str:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    response = requests.get(f"{_EUTILS_BASE}/efetch.fcgi", params=params, timeout=60)
    response.raise_for_status()
    time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS)
    # Strip the XML declaration/root wrapper from each batch - the caller wraps all
    # batches in a single <PubmedArticleSet> root instead.
    text = response.text
    start = text.find("<PubmedArticle>")
    end = text.rfind("</PubmedArticle>") + len("</PubmedArticle>")
    return text[start:end] if start != -1 and end != -1 else ""
