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
    return hashlib.sha1(
        normalize_for_match(text).encode("utf-8"), usedforsecurity=False
    ).hexdigest()


def _clean_markdown(md: str) -> str:
    """Drop images and reduce links to their text so the model never sees URL syntax (fewer
    tokens, and quotes copied from excerpts stay verbatim relative to the page)."""
    md = _IMAGE_RE.sub("", md)
    md = _LINK_RE.sub(r"\1", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


_NAV_LINE_RE = re.compile(r"^\s*(?:[-*+•]\s*)?[^.!?]{1,40}$")


def _is_navigational(paragraph: str) -> bool:
    """Very short items, or blocks made of many short sentence-less lines (menus, footers,
    cookie banners) → navigation noise."""
    if len(paragraph) < MIN_CHUNK_CHARS:
        return True
    lines = [ln for ln in paragraph.splitlines() if ln.strip()]
    return bool(
        len(lines) >= 4 and sum(1 for ln in lines if _NAV_LINE_RE.match(ln)) >= 0.75 * len(lines)
    )


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


def paragraph_hashes(markdown: str) -> set[str]:
    md = _clean_markdown(markdown or "")
    body = "\n".join(line for line in md.splitlines() if not _HEADING_RE.match(line.strip()))
    return {
        text_hash(p)
        for p in re.split(r"\n\s*\n", body)
        if p.strip() and len(p.strip()) >= MIN_CHUNK_CHARS
    }


def chunk_markdown(
    markdown: str,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
    page_title: str = "",
    known_paragraphs: set[str] | None = None,
) -> tuple[list[Chunk], int]:
    """Chunk a page. Paragraphs whose hash is in `known_paragraphs` (seen on other pages:
    nav, footers, cookie banners) are dropped. Returns (chunks, dropped_paragraphs)."""
    known = known_paragraphs or set()
    dropped = 0
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
        fresh = [p for p in paragraphs if text_hash(p) not in known]
        dropped += len(paragraphs) - len(fresh)
        paragraphs = fresh
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
    return chunks, dropped
