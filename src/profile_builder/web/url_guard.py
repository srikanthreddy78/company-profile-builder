"""URL validation and normalization (SSRF protection).

Rules: http(s) only; no userinfo; default ports only; DNS-resolved addresses must be public
(no loopback / private / link-local / CGNAT / multicast / reserved / unspecified, IPv4 or IPv6);
discovered pages must stay on the start site; URLs are normalized so duplicates collapse.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from profile_builder.config import MAX_URL_LENGTH

ALLOWED_SCHEMES = frozenset({"http", "https"})
DEFAULT_PORTS = {"http": 80, "https": 443}
TRACKING_PARAMS = re.compile(r"^(utm_[a-z]+|fbclid|gclid|mc_cid|mc_eid|ref|source)$", re.I)

Resolver = Callable[[str], list[str]]


class URLGuardError(ValueError):
    """Raised when a URL must not be fetched."""


def _default_resolver(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError, ValueError) as exc:
        raise URLGuardError(f"DNS resolution failed for {host!r}: {exc}") from exc
    return sorted({info[4][0] for info in infos})


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    if isinstance(ip, ipaddress.IPv4Address):
        if ip in ipaddress.ip_network("100.64.0.0/10"):  # CGNAT
            return False
        if ip in ipaddress.ip_network("0.0.0.0/8"):
            return False
    else:
        if ip in ipaddress.ip_network("fc00::/7") or ip in ipaddress.ip_network("fe80::/10"):
            return False
        if ip.sixtofour is not None or ip.teredo is not None:
            return False
    return True


# Multi-tenant hosting suffixes where each subdomain belongs to a different owner.
MULTI_TENANT_SUFFIXES = frozenset(
    {
        "github.io",
        "gitlab.io",
        "vercel.app",
        "netlify.app",
        "herokuapp.com",
        "pages.dev",
        "web.app",
        "firebaseapp.com",
        "azurewebsites.net",
        "cloudfront.net",
        "amazonaws.com",
        "blogspot.com",
        "wordpress.com",
        "wixsite.com",
        "squarespace.com",
        "webflow.io",
        "myshopify.com",
        "readthedocs.io",
        "notion.site",
        "framer.website",
        "fly.dev",
        "onrender.com",
    }
)


def registrable_domain(host: str) -> str:
    """Approximate eTLD+1 without a public-suffix list: common two-part TLDs and well-known
    multi-tenant hosts are handled; IP literals are compared exactly."""
    host = (host or "").lower().strip(".")
    try:
        return str(ipaddress.ip_address(host.strip("[]")))
    except ValueError:
        pass
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    if ".".join(parts[-2:]) in MULTI_TENANT_SUFFIXES:
        return ".".join(parts[-3:])
    second_level = {"co", "com", "org", "net", "gov", "edu", "ac"}
    if parts[-2] in second_level and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def normalize_url(url: str) -> str:
    """Lowercase scheme/host, drop fragment, default ports, tracking params; sort query."""
    try:
        parts = urlsplit((url or "").strip())
        port = parts.port
    except ValueError as exc:
        raise URLGuardError(f"unparseable URL: {exc}") from exc
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower().rstrip(".")
    netloc = host
    if port and port != DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path) or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not TRACKING_PARAMS.match(k)
    ]
    query = urlencode(sorted(query_pairs))
    return urlunsplit((scheme, netloc, path, query, ""))


def validate_url(url: str, *, resolver: Resolver | None = None, check_dns: bool = True) -> str:
    """Validate and normalize a URL. Raises URLGuardError if it must not be fetched."""
    try:
        return _validate_url(url, resolver=resolver, check_dns=check_dns)
    except URLGuardError:
        raise
    except (ValueError, UnicodeError) as exc:  # hostile input must never crash the caller
        raise URLGuardError(f"invalid URL: {exc}") from exc


def _validate_url(url: str, *, resolver: Resolver | None, check_dns: bool) -> str:
    if not url or not isinstance(url, str):
        raise URLGuardError("empty URL")
    if len(url) > MAX_URL_LENGTH:
        raise URLGuardError("URL too long")
    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        raise URLGuardError(f"unparseable URL: {exc}") from exc
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise URLGuardError(f"scheme {scheme or '(none)'!r} not allowed; use http or https")
    if parts.username is not None or parts.password is not None:
        raise URLGuardError("URLs with embedded credentials are not allowed")
    host = parts.hostname
    if not host:
        raise URLGuardError("URL has no host")
    host = host.lower().rstrip(".")
    try:
        port = parts.port
    except ValueError as exc:
        raise URLGuardError("invalid port") from exc
    if port is not None and port != DEFAULT_PORTS[scheme]:
        raise URLGuardError(f"non-default port {port} is not allowed")
    if host in {"localhost", "localhost.localdomain", "ip6-localhost"} or host.endswith(
        ".localhost"
    ):
        raise URLGuardError("localhost is not allowed")
    if host.endswith((".local", ".internal", ".lan", ".home", ".corp", ".localdomain")):
        raise URLGuardError(f"internal hostname {host!r} is not allowed")

    # Literal IP (any notation ipaddress accepts, incl. IPv6 in brackets)
    literal_ip: str | None = None
    try:
        literal_ip = str(ipaddress.ip_address(host))
    except ValueError:
        # Decimal/hex/octal IP notations (2130706433, 0x7f000001, 0177.0.0.1, 127.1): every
        # label looks like a number. Real hostnames such as "bad.be" are unaffected.
        if all(re.fullmatch(r"0x[0-9a-f]+|0[0-7]*|[1-9][0-9]*", lab) for lab in host.split(".")):
            raise URLGuardError(f"numeric host {host!r} is not allowed") from None
    if literal_ip is not None:
        if not _is_public_ip(literal_ip):
            raise URLGuardError(f"IP address {literal_ip} is not publicly routable")
        return normalize_url(url)

    if check_dns:
        resolve = resolver or _default_resolver
        addresses = resolve(host)
        if not addresses:
            raise URLGuardError(f"host {host!r} did not resolve")
        for addr in addresses:
            if not _is_public_ip(addr):
                raise URLGuardError(
                    f"host {host!r} resolves to non-public address {addr}; refusing to fetch"
                )
    return normalize_url(url)


def same_site(url: str, start_url: str) -> bool:
    """Same registrable domain, and no https → http downgrade relative to the start URL."""
    try:
        u, b = urlsplit(url), urlsplit(start_url)
    except ValueError:
        return False
    if b.scheme == "https" and u.scheme == "http":
        return False
    return registrable_domain(u.hostname or "") == registrable_domain(b.hostname or "")
