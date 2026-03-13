"""
arxiv_client.py
Thin wrapper around the arXiv Atom API.

arXiv API docs: https://arxiv.org/help/api/user-manual
"""

from __future__ import annotations

import logging
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Tuple

from .document import Document
from .robots import checker as _robots

# ---------------------------------------------------------------------------
# SSL context — tries to use certifi if installed, otherwise falls back to
# an unverified context (acceptable for a research crawler on a local machine).
# ---------------------------------------------------------------------------
def _build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except ImportError:
        pass
    # Fallback: disable verification (Windows often lacks the system CA bundle)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

_SSL_CTX = _build_ssl_context()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

# AI / ML arXiv categories
AI_ML_CATEGORIES = [
    "cs.AI",    # Artificial Intelligence
    "cs.LG",    # Machine Learning
    "cs.CV",    # Computer Vision
    "cs.CL",    # Computation & Language (NLP)
    "cs.NE",    # Neural and Evolutionary Computing
    "stat.ML",  # Statistics – Machine Learning
]

DEFAULT_SEARCH_QUERY = " OR ".join(f"cat:{c}" for c in AI_ML_CATEGORIES)

# Polite delay between requests (arXiv asks for ≥ 3 s between calls)
REQUEST_DELAY = 3.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tag(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def _get_text(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    return (element.text or "").strip()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class ArxivClient:
    """Fetches article IDs and full metadata from the arXiv Atom API."""

    def __init__(
        self,
        search_query: str = DEFAULT_SEARCH_QUERY,
        request_delay: float = REQUEST_DELAY,
        timeout: int = 30,
    ) -> None:
        self.search_query = search_query
        self.request_delay = request_delay
        self.timeout = timeout
        self._last_request_time: float = 0.0

    # -----------------------------------------------------------------------
    # Rate-limited HTTP helper
    # -----------------------------------------------------------------------
    def _get(self, url: str) -> str:
        # ── robots.txt compliance ────────────────────────────────────────────
        if not _robots.allowed(url):
            raise PermissionError(f"robots.txt disallows fetching: {url}")

        # ── Rate limiting: respect robots.txt Crawl-delay + our own minimum ─
        robots_delay = _robots.crawl_delay(url)
        effective_delay = max(self.request_delay, robots_delay)
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < effective_delay:
            time.sleep(effective_delay - elapsed)

        logger.debug("GET %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "SRI-Crawler/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_CTX) as resp:
            content = resp.read().decode("utf-8")
        self._last_request_time = time.monotonic()
        return content

    # -----------------------------------------------------------------------
    # ID discovery
    # -----------------------------------------------------------------------
    def fetch_ids(
        self,
        max_results: int = 100,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> List[str]:
        """
        Query arXiv and return a list of article IDs (without version suffix).

        Parameters
        ----------
        max_results : int
            Number of results to request per API call (max 2000).
        start : int
            Pagination offset.
        sort_by : str
            'submittedDate' | 'lastUpdatedDate' | 'relevance'
        sort_order : str
            'ascending' | 'descending'
        """
        params = urllib.parse.urlencode(
            {
                "search_query": self.search_query,
                "start": start,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }
        )
        url = f"{BASE_URL}?{params}"

        try:
            xml_text = self._get(url)
        except Exception as exc:
            logger.error("fetch_ids failed: %s", exc)
            return []

        return self._parse_ids(xml_text)

    def _parse_ids(self, xml_text: str) -> List[str]:
        root = ET.fromstring(xml_text)
        ids: List[str] = []
        for entry in root.findall(_tag(ATOM_NS, "entry")):
            id_elem = entry.find(_tag(ATOM_NS, "id"))
            if id_elem is not None and id_elem.text:
                # URL looks like  https://arxiv.org/abs/2301.12345v2
                raw = id_elem.text.strip().rstrip("/")
                arxiv_id = raw.split("/abs/")[-1]
                # Strip version suffix (v1, v2, …)
                arxiv_id = arxiv_id.split("v")[0]
                ids.append(arxiv_id)
        return ids

    # -----------------------------------------------------------------------
    # Full metadata fetch
    # -----------------------------------------------------------------------
    def fetch_documents(self, arxiv_ids: List[str]) -> List[Document]:
        """
        Fetch full metadata for the given IDs and return Document objects.

        arXiv allows up to ~20 IDs per request via the `id_list` parameter.
        We chunk automatically to respect that limit.
        """
        if not arxiv_ids:
            return []

        docs: List[Document] = []
        chunk_size = 20
        for i in range(0, len(arxiv_ids), chunk_size):
            chunk = arxiv_ids[i : i + chunk_size]
            docs.extend(self._fetch_chunk(chunk))
        return docs

    def _fetch_chunk(self, arxiv_ids: List[str]) -> List[Document]:
        params = urllib.parse.urlencode(
            {
                "id_list": ",".join(arxiv_ids),
                "max_results": len(arxiv_ids),
            }
        )
        url = f"{BASE_URL}?{params}"

        try:
            xml_text = self._get(url)
        except Exception as exc:
            logger.error("fetch_documents chunk failed: %s", exc)
            return []

        return self._parse_entries(xml_text)

    def _parse_entries(self, xml_text: str) -> List[Document]:
        root = ET.fromstring(xml_text)
        docs: List[Document] = []

        for entry in root.findall(_tag(ATOM_NS, "entry")):
            try:
                doc = self._entry_to_document(entry)
                if doc:
                    docs.append(doc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not parse entry: %s", exc)

        return docs

    def _entry_to_document(self, entry: ET.Element) -> Optional[Document]:
        # ID
        id_elem = entry.find(_tag(ATOM_NS, "id"))
        if id_elem is None or not id_elem.text:
            return None
        raw_id = id_elem.text.strip().rstrip("/").split("/abs/")[-1]
        arxiv_id = raw_id.split("v")[0]

        # Title
        title = _get_text(entry.find(_tag(ATOM_NS, "title"))).replace("\n", " ")

        # Authors
        authors = ", ".join(
            _get_text(a.find(_tag(ATOM_NS, "name")))
            for a in entry.findall(_tag(ATOM_NS, "author"))
        )

        # Abstract
        abstract = _get_text(entry.find(_tag(ATOM_NS, "summary"))).replace("\n", " ")

        # Categories
        cats = [
            c.get("term", "")
            for c in entry.findall(_tag(ATOM_NS, "category"))
        ]
        categories = ", ".join(filter(None, cats))

        # Dates
        published = _get_text(entry.find(_tag(ATOM_NS, "published")))
        updated = _get_text(entry.find(_tag(ATOM_NS, "updated")))

        # PDF link
        pdf_url = ""
        for link in entry.findall(_tag(ATOM_NS, "link")):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        return Document(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            abstract=abstract,
            categories=categories,
            published=published,
            updated=updated,
            pdf_url=pdf_url,
        )