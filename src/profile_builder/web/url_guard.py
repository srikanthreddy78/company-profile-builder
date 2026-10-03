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
    except socket.gaierror as exc:
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


def registrable_domain(host: str) -> str:
    """Approximate eTLD+1 without a public-suffix list: handles common two-part TLDs."""
    parts = host.lower().strip(".").split(".")
    if len(parts) <= 2:
        return ".".join(parts)
    second_level = {"co", "com", "org", "net", "gov", "edu", "ac"}
    if parts[-2] in second_level and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def normalize_url(url: str) -> str:
    """Lowercase scheme/host, drop fragment, default ports, tracking params; sort query."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower().rstrip(".")
    port = parts.port
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
        if re.fullmatch(r"[0-9a-fx.]+", host) and not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
            # Decimal/hex/octal-looking host like 2130706433 or 0x7f000001 or 0177.0.0.1
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
    a = urlsplit(url).hostname or ""
    b = urlsplit(start_url).hostname or ""
    return registrable_domain(a) == registrable_domain(b)
