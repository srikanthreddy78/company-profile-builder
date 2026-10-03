"""Heading-aware markdown chunking with cross-page de-duplication support."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from profile_builder.config import CHARS_PER_TOKEN, CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS
from profile_builder.security import normalize_for_match

MAX_CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN
MIN_CHUNK_CHARS = 40

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    ordinal: int
    heading: str
    text: str
    text_hash: str


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_for_match(text).encode("utf-8"), usedforsecurity=False).hexdigest()


def _clean_markdown(md: str) -> str:
    md = _IMAGE_RE.sub("", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _is_navigational(paragraph: str) -> bool:
    """Mostly links / very short items → navigation, footer, cookie banners."""
    links = _LINK_RE.findall(paragraph)
    link_text = sum(len(t) for t in links)
    plain = _LINK_RE.sub("", paragraph)
    if len(paragraph) < MIN_CHUNK_CHARS:
        return True
    return bool(links) and link_text > 0.6 * max(len(plain) + link_text, 1)


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    sentences = _SENTENCE_RE.split(text)
    pieces: list[str] = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = f"{current} {s}".strip()
            continue
        if current:
            pieces.append(current)
        if len(s) > max_chars:  # single giant sentence → hard split by words
            words = s.split()
            buf = ""
            for w in words:
                if len(buf) + len(w) + 1 > max_chars:
                    pieces.append(buf)
                    buf = buf[-overlap:] if overlap else ""
                buf = f"{buf} {w}".strip()
            current = buf
        else:
            tail = pieces[-1][-overlap:] if pieces and overlap else ""
            current = f"{tail} {s}".strip() if tail else s
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(
    markdown: str,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
    page_title: str = "",
) -> list[Chunk]:
    md = _clean_markdown(markdown or "")
    heading = page_title or ""
    sections: list[tuple[str, list[str]]] = [(heading, [])]
    for line in md.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            heading = m.group(2).strip()
            sections.append((heading, []))
        else:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    seen_in_page: set[str] = set()
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        paragraphs = [p for p in paragraphs if not _is_navigational(p)]
        buf = ""
        pending: list[str] = []
        for p in paragraphs:
            if len(buf) + len(p) + 2 <= max_chars:
                buf = f"{buf}\n\n{p}".strip()
                continue
            if buf:
                pending.append(buf)
            buf = p if len(p) <= max_chars else ""
            if len(p) > max_chars:
                pending.extend(_split_long(p, max_chars, overlap))
        if buf:
            pending.append(buf)
        for text in pending:
            if len(text) < MIN_CHUNK_CHARS:
                continue
            h = text_hash(text)
            if h in seen_in_page:
                continue
            seen_in_page.add(h)
            chunks.append(Chunk(ordinal=len(chunks), heading=heading[:120], text=text, text_hash=h))
    return chunks
