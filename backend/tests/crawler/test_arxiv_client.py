"""Tests de ArxivClient — rate limit, parseo XML, texto y retrocompat."""
from __future__ import annotations
import threading
import time
import xml.etree.ElementTree as ET
import pytest

ATOM_NS = "http://www.w3.org/2005/Atom"


def _make_entry(arxiv_id="2301.12345v2", title="Test", author="Alice",
                abstract="Abs", categories=("cs.AI",),
                pdf_href="https://arxiv.org/pdf/2301.12345"):
    entry_str = f"""<entry xmlns="{ATOM_NS}">
      <id>https://arxiv.org/abs/{arxiv_id}</id>
      <title>{title}</title>
      <author><n>{author}</n></author>
      <summary>{abstract}</summary>
      {"".join(f'<category term="{c}"/>' for c in categories)}
      <published>2023-01-01T00:00:00Z</published>
      <updated>2023-01-02T00:00:00Z</updated>
      <link title="pdf" href="{pdf_href}"/>
    </entry>"""
    return ET.fromstring(entry_str)


# ── Política de crawling ────────────────────────────────────────────────────

def test_source_name():
    from backend.crawler.clients import ArxivClient
    assert ArxivClient().source_name == "arxiv"


def test_request_delay_at_least_15_seconds():
    from backend.crawler.clients import ArxivClient
    assert ArxivClient().request_delay >= 15.0


def test_trusted_domains_contains_arxiv_hosts():
    from backend.crawler.clients import ArxivClient
    td = ArxivClient().trusted_domains
    assert isinstance(td, frozenset)
    assert "arxiv.org" in td
    assert "export.arxiv.org" in td


def test_allowed_uses_trusted_domains():
    import urllib.robotparser
    from backend.crawler.robots import RobotsChecker
    from backend.crawler.clients import ArxivClient

    c = ArxivClient()
    rc = RobotsChecker()
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(["User-agent: *", "Disallow: /api", "Crawl-delay: 15"])
    rc._cache["https://export.arxiv.org"] = (parser, time.monotonic())

    assert rc.allowed("https://export.arxiv.org/api/query") is False
    assert rc.allowed("https://export.arxiv.org/api/query", c.trusted_domains) is True


def test_crawl_delay_not_bypassed_by_trusted_domains():
    import urllib.robotparser
    from backend.crawler.robots import RobotsChecker

    rc = RobotsChecker()
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(["User-agent: *", "Crawl-delay: 15", "Disallow: /api"])
    for origin in ["https://arxiv.org", "https://export.arxiv.org"]:
        rc._cache[origin] = (parser, time.monotonic())

    assert rc.crawl_delay("https://arxiv.org/pdf/2301.12345") == 15.0


# ── Parseo XML (a través del módulo api) ────────────────────────────────────

def test_make_doc_id():
    from backend.crawler.clients import ArxivClient
    assert ArxivClient().make_doc_id("2301.12345") == "arxiv:2301.12345"


def test_parse_ids_strips_version():
    from backend.crawler.clients.arxiv import api
    xml = f"""<feed xmlns="{ATOM_NS}">
      <entry><id>https://arxiv.org/abs/2301.00001v3</id></entry>
      <entry><id>https://arxiv.org/abs/2302.99999v1</id></entry>
    </feed>"""
    assert api.parse_ids(xml) == ["2301.00001", "2302.99999"]


def test_entry_to_document_produces_composite_doc_id():
    from backend.crawler.clients.arxiv import api
    from backend.crawler.clients import ArxivClient
    make_doc_id = ArxivClient().make_doc_id
    doc = api._entry_to_document(_make_entry(), make_doc_id)
    assert doc is not None
    assert doc.doc_id == "arxiv:2301.12345"


def test_entry_without_id_returns_none():
    from backend.crawler.clients.arxiv import api
    from backend.crawler.clients import ArxivClient
    entry = ET.fromstring(f'<entry xmlns="{ATOM_NS}"><title>No ID</title></entry>')
    assert api._entry_to_document(entry, ArxivClient().make_doc_id) is None


def test_fetch_documents_groups_into_chunks_of_20():
    from backend.crawler.clients import ArxivClient
    client = ArxivClient()
    sizes = []
    client._fetch_chunk = lambda ids: (sizes.append(len(ids)) or [])
    client.fetch_documents([f"{i:04d}" for i in range(55)])
    assert sizes == [20, 20, 15]


# ── Rate limiting ───────────────────────────────────────────────────────────

def test_rate_limit_is_thread_safe_single_instance():
    from backend.crawler.clients import ArxivClient

    class FastClient(ArxivClient):
        @property
        def request_delay(self): return 0.05

    client = FastClient()
    ArxivClient._last_request = 0.0
    times = []

    def simulate_get():
        with ArxivClient._rate_lock:
            elapsed = time.monotonic() - ArxivClient._last_request
            if elapsed < client.request_delay:
                time.sleep(client.request_delay - elapsed)
            ArxivClient._last_request = time.monotonic()
            times.append(ArxivClient._last_request)

    threads = [threading.Thread(target=simulate_get) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    times.sort()
    gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
    assert all(g >= 0.045 for g in gaps)


def test_rate_limit_shared_across_two_instances():
    from backend.crawler.clients import ArxivClient

    class FastClient(ArxivClient):
        @property
        def request_delay(self): return 0.05

    a1, a2 = FastClient(), FastClient()
    assert a1._rate_lock is a2._rate_lock
    ArxivClient._last_request = 0.0
    times = []

    def call(client):
        with ArxivClient._rate_lock:
            elapsed = time.monotonic() - ArxivClient._last_request
            if elapsed < client.request_delay:
                time.sleep(client.request_delay - elapsed)
            ArxivClient._last_request = time.monotonic()
            times.append(ArxivClient._last_request)

    t1 = threading.Thread(target=call, args=(a1,))
    t2 = threading.Thread(target=call, args=(a2,))
    t1.start(); t2.start()
    t1.join(); t2.join()

    times.sort()
    assert times[1] - times[0] >= 0.045


# ── Extracción de texto ─────────────────────────────────────────────────────

def test_latexml_parser_skips_bibliography():
    from backend.crawler.clients.arxiv.extractors.html import _LaTeXMLParser
    html = """<div class="ltx_document">
      <p class="ltx_p">Main content here.</p>
      <section class="ltx_bibliography"><p>References.</p></section>
    </div>"""
    p = _LaTeXMLParser()
    p.feed(html)
    text = p.result()
    assert "Main content" in text
    assert "References" not in text


def test_latexml_parser_skips_authors():
    from backend.crawler.clients.arxiv.extractors.html import _LaTeXMLParser
    html = """<div class="ltx_document">
      <div class="ltx_authors">Alice, Bob</div>
      <p class="ltx_p">Abstract text here.</p>
    </div>"""
    p = _LaTeXMLParser()
    p.feed(html)
    text = p.result()
    assert "Abstract text" in text
    assert "Alice, Bob" not in text


def test_html_extract_returns_nonempty_for_valid_html():
    from backend.crawler.clients.arxiv.extractors.html import extract
    # Minimal ltx_document so it passes the 500-char threshold
    content = "This is test content. " * 30
    html = f"""<html><body>
      <div class="ltx_document">
        <p class="ltx_p">{content}</p>
      </div>
    </body></html>""".encode()
    text = extract(html)
    assert len(text) > 0
    assert "test content" in text


def test_pdf_extractor_module_does_not_exist():
    import importlib, pathlib
    path = pathlib.Path("backend/crawler/pdf_extractor.py")
    assert not path.exists()
    with pytest.raises((ImportError, ModuleNotFoundError)):
        importlib.import_module("backend.crawler.pdf_extractor")


# ── Retrocompat ─────────────────────────────────────────────────────────────

def test_backward_compat_import():
    from backend.crawler import ArxivClient
    c = ArxivClient()
    assert c.source_name == "arxiv"
    assert c.request_delay >= 15.0


@pytest.mark.network
def test_fetch_ids_returns_local_ids():
    from backend.crawler.clients import ArxivClient
    ids = ArxivClient().fetch_ids(max_results=3)
    assert len(ids) > 0
    for lid in ids:
        assert ":" not in lid
        assert "v" not in lid


@pytest.mark.network
def test_fetch_documents_returns_composite_doc_ids():
    from backend.crawler.clients import ArxivClient
    c = ArxivClient()
    docs = c.fetch_documents(c.fetch_ids(max_results=2))
    for doc in docs:
        assert doc.doc_id.startswith("arxiv:")