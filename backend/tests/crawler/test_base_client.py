"""Tests de BaseClient — interfaz abstracta, make_doc_id, parse_doc_id."""
from __future__ import annotations
import pytest


def _minimal_client(source="mysource", delay=5.0, trusted=frozenset()):
    from backend.crawler.clients.base_client import BaseClient

    class MinClient(BaseClient):
        @property
        def source_name(self): return source
        @property
        def request_delay(self): return delay
        @property
        def trusted_domains(self): return trusted
        def fetch_ids(self, **kw): return []
        def fetch_documents(self, ids): return []
        def download_text(self, lid, **kw): return ""

    return MinClient()


def test_cannot_instantiate_abstract_class():
    from backend.crawler.clients.base_client import BaseClient
    with pytest.raises(TypeError):
        BaseClient()


def test_missing_request_delay_raises():
    from backend.crawler.clients.base_client import BaseClient
    with pytest.raises(TypeError):
        class Incomplete(BaseClient):
            @property
            def source_name(self): return "x"
            @property
            def trusted_domains(self): return frozenset()
            def fetch_ids(self, **kw): return []
            def fetch_documents(self, ids): return []
            def download_text(self, lid, **kw): return ""
        Incomplete()


def test_missing_trusted_domains_raises():
    from backend.crawler.clients.base_client import BaseClient
    with pytest.raises(TypeError):
        class Incomplete(BaseClient):
            @property
            def source_name(self): return "x"
            @property
            def request_delay(self): return 5.0
            def fetch_ids(self, **kw): return []
            def fetch_documents(self, ids): return []
            def download_text(self, lid, **kw): return ""
        Incomplete()


def test_request_delay_is_float():
    c = _minimal_client(delay=10.0)
    assert isinstance(c.request_delay, float)
    assert c.request_delay == 10.0


def test_trusted_domains_is_frozenset():
    c = _minimal_client(trusted=frozenset({"a.com", "b.com"}))
    assert isinstance(c.trusted_domains, frozenset)
    assert "a.com" in c.trusted_domains


def test_make_doc_id_format():
    c = _minimal_client()
    assert c.make_doc_id("abc123") == "mysource:abc123"


def test_parse_doc_id_valid():
    from backend.crawler.clients.base_client import BaseClient
    source, local = BaseClient.parse_doc_id("arxiv:2301.12345")
    assert source == "arxiv" and local == "2301.12345"


def test_parse_doc_id_preserves_colons_in_local():
    from backend.crawler.clients.base_client import BaseClient
    source, local = BaseClient.parse_doc_id("fake:id:with:colons")
    assert source == "fake" and local == "id:with:colons"


@pytest.mark.parametrize("bad_id", ["nocolon", ":noleft", "noright:", ""])
def test_parse_doc_id_invalid_raises(bad_id):
    from backend.crawler.clients.base_client import BaseClient
    with pytest.raises(ValueError):
        BaseClient.parse_doc_id(bad_id)


def test_make_and_parse_are_inverse():
    from backend.crawler.clients.base_client import BaseClient
    c = _minimal_client(source="src")
    for lid in ["id1", "complex.id.v3"]:
        composite = c.make_doc_id(lid)
        src, parsed = BaseClient.parse_doc_id(composite)
        assert src == "src" and parsed == lid
