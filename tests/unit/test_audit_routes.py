"""Tests for development/production API proxy parity checks."""

from scripts.tools.audit_routes import get_nginx_proxies


def test_nginx_proxy_covers_bare_and_nested_api_paths():
    prefixes, matches_bare = get_nginx_proxies()
    assert matches_bare is True
    assert "/jobs" in prefixes
    assert "/backtest-management" in prefixes
    assert "/forecast-release" in prefixes
    assert "/ai-champion" in prefixes
    assert "/expsys" not in prefixes
