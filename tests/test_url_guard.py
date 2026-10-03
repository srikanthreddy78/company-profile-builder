"""SSRF guard: nothing private, local or non-HTTP is ever fetched."""

from __future__ import annotations

import pytest

from profile_builder.web.url_guard import (
    URLGuardError,
    normalize_url,
    registrable_domain,
    same_site,
    validate_url,
)

PUBLIC = ["93.184.216.34"]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "javascript:alert(1)",
        "http://localhost/",
        "http://LOCALHOST:80/",
        "http://foo.localhost/",
        "http://intranet.local/",
        "http://127.0.0.1/",
        "http://127.1/",
        "http://0.0.0.0/",
        "http://10.1.2.3/",
        "http://172.16.0.9/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://100.64.0.1/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fd00::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://2130706433/",
        "http://0x7f000001/",
        "http://0177.0.0.1/",
        "http://user:pw@example.com/",
        "http://example.com:8080/",
        "https://example.com:8443/",
        "",
        "http:///path",
        "http://" + "a" * 3000 + ".com/",
    ],
)
def test_rejected(url):
    with pytest.raises(URLGuardError):
        validate_url(url, resolver=lambda h: PUBLIC)


def test_dns_rebinding_to_private_is_rejected():
    with pytest.raises(URLGuardError, match="non-public"):
        validate_url("https://evil.example.com/", resolver=lambda h: ["93.184.216.34", "10.0.0.5"])
    with pytest.raises(URLGuardError):
        validate_url("https://evil.example.com/", resolver=lambda h: ["::1"])


def test_public_allowed_and_normalized():
    out = validate_url(
        "HTTPS://Example.com:443/a//b/?utm_source=x&b=2&a=1#frag", resolver=lambda h: PUBLIC
    )
    assert out == "https://example.com/a/b?a=1&b=2"
    assert validate_url("http://example.com", resolver=lambda h: PUBLIC) == "http://example.com/"
    assert (
        validate_url("https://93.184.216.34/x", resolver=lambda h: PUBLIC)
        == "https://93.184.216.34/x"
    )


def test_normalize_dedupes_variants():
    a = normalize_url("https://www.fortanix.com/platform/")
    b = normalize_url("https://www.fortanix.com/platform?utm_campaign=x#top")
    assert a == b == "https://www.fortanix.com/platform"


def test_same_site_and_registrable_domain():
    assert registrable_domain("www.fortanix.com") == "fortanix.com"
    assert registrable_domain("docs.example.co.uk") == "example.co.uk"
    assert same_site("https://docs.fortanix.com/x", "https://www.fortanix.com/")
    assert not same_site("https://fortanix.com.evil.net/", "https://www.fortanix.com/")
