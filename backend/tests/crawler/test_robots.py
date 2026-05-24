"""Tests de RobotsChecker — trusted_domains, allow/deny, crawl_delay."""
from __future__ import annotations
import time
import urllib.robotparser
import pytest


def _inject_parser(rc, origin, rules):
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(rules)
    rc._cache[origin] = (parser, time.monotonic())


def test_no_allowed_domains_global():
    import backend.crawler.robots as robots_mod
    assert not hasattr(robots_mod, "ALLOWED_DOMAINS")


def test_allowed_without_trusted_reads_robots():
    from backend.crawler.robots import RobotsChecker
    rc = RobotsChecker()
    _inject_parser(rc, "https://example.com", ["User-agent: *", "Disallow: /secret"])
    assert rc.allowed("https://example.com/public") is True
    assert rc.allowed("https://example.com/secret") is False


def test_trusted_domains_bypasses_disallow():
    from backend.crawler.robots import RobotsChecker
    rc = RobotsChecker()
    _inject_parser(rc, "https://api.example.com", ["User-agent: *", "Disallow: /api"])
    assert rc.allowed("https://api.example.com/api/query") is False
    assert rc.allowed("https://api.example.com/api/query",
                      trusted_domains=frozenset({"api.example.com"})) is True


def test_crawl_delay_never_bypassed_by_trusted_domains():
    from backend.crawler.robots import RobotsChecker
    rc = RobotsChecker()
    _inject_parser(rc, "https://api.example.com",
                   ["User-agent: *", "Crawl-delay: 15", "Disallow: /api"])
    delay = rc.crawl_delay("https://api.example.com/api/query")
    assert delay == 15.0


def test_crawl_delay_zero_when_not_declared():
    from backend.crawler.robots import RobotsChecker
    rc = RobotsChecker()
    _inject_parser(rc, "https://nodelay.example.com", ["User-agent: *", "Disallow: /private"])
    assert rc.crawl_delay("https://nodelay.example.com/page") == 0.0


def test_allowed_fail_open_on_network_error():
    from backend.crawler.robots import RobotsChecker
    rc = RobotsChecker(ttl=0)
    result = rc.allowed("https://this-domain-does-not-exist-xyz123.invalid/page")
    assert result is True
