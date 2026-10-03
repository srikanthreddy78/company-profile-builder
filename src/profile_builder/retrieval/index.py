"""Hybrid retrieval over scraped chunks: BM25 + embedding cosine fused with RRF.

Chunks and embeddings live in the RunStore (SQLite), so the index survives restarts and
is rebuilt lazily in memory per process.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from profile_builder.config import (
    CHARS_PER_TOKEN,
    MAX_EXCERPT_CHARS,
    MAX_READ_CHARS,
    RRF_K,
    SEARCH_K,
)
from profile_builder.logging_setup import get_logger
from profile_builder.models import estimate_cost_usd
from profile_builder.retrieval.chunking import chunk_markdown, paragraph_hashes
from profile_builder.state.run_store import ChunkRecord, RunStore

log = get_logger("index")

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']+")
_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "by",
        "at",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "we",
        "our",
        "you",
        "your",
        "they",
        "their",
        "he",
        "she",
        "his",
        "her",
        "them",
        "us",
        "i",
        "me",
        "my",
        "can",
        "will",
        "may",
    ]
)


class Embeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddings:
    """Deterministic, dependency-free embeddings for tests and `--no-embeddings` fallback."""

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dims, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
            vec[h % self.dims] += 1.0 if (h >> 8) % 2 else -1.0
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class SearchHit:
    url: str
    heading: str
    excerpt: str
    ordinal: int
    score: float


class HybridIndex:
    def __init__(
        self,
        store: RunStore,
        embeddings: Embeddings | None,
        *,
        embedding_model: str = "",
        use_embeddings: bool = True,
    ) -> None:
        self.store = store
        self.embeddings = embeddings if use_embeddings else None
        self.embedding_model = embedding_model
        self._chunks: list[ChunkRecord] | None = None
        self._token_sets: list[set[str]] = []
        self._bm25: BM25Okapi | None = None
        self._matrix: np.ndarray | None = None
        self._matrix_rows: list[int] = []  # chunk positions that have an embedding
        self.last_embedding_error: str | None = None  # set by the last index_page call

    # ---- indexing ------------------------------------------------------------------
    def index_page(self, url: str, markdown: str, *, title: str = "") -> tuple[int, int]:
        """Chunk + store a page, dropping paragraphs already seen on *other* pages (shared
        nav/footers) and chunks whose text repeats elsewhere. Returns (chunks_kept, dropped).
        An embedding-provider failure never aborts indexing: the page is stored BM25-only and
        `last_embedding_error` carries the reason for the caller to warn about."""
        known = self.store.paragraph_hashes(exclude_url=url)
        chunks, dropped_paragraphs = chunk_markdown(
            markdown, page_title=title, known_paragraphs=known
        )
        foreign_chunks = self.store.chunk_hashes(exclude_url=url)
        kept = [c for c in chunks if c.text_hash not in foreign_chunks]
        dropped = dropped_paragraphs + (len(chunks) - len(kept))
        vectors: list[np.ndarray | None] = [None] * len(kept)
        self.last_embedding_error = None
        if self.embeddings is not None and kept:
            texts = [c.text for c in kept]
            try:
                embedded = self.embeddings.embed_documents(texts)
                vectors = [np.asarray(v, dtype=np.float32) for v in embedded]
            except Exception as exc:
                self.last_embedding_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning("embedding failed for %s; keyword index only: %s", url, exc)
            else:
                approx_tokens = sum(len(t) for t in texts) // CHARS_PER_TOKEN
                self.store.add_usage(
                    "embedding",
                    self.embedding_model,
                    approx_tokens,
                    0,
                    estimate_cost_usd(self.embedding_model, {"input_tokens": approx_tokens}),
                )
        self.store.replace_chunks(
            url, [(i, c.heading, c.text, c.text_hash, vectors[i]) for i, c in enumerate(kept)]
        )
        self.store.replace_paragraphs(url, paragraph_hashes(markdown))
        self._invalidate()
        return len(kept), dropped

    def _invalidate(self) -> None:
        self._chunks = None
        self._bm25 = None
        self._matrix = None

    def _load(self) -> list[ChunkRecord]:
        if self._chunks is None:
            self._chunks = self.store.list_chunks()
            corpus = [tokenize(c.text) or ["_"] for c in self._chunks]
            self._token_sets = [set(toks) for toks in corpus]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            # Pages indexed while the embedding provider was down have no vectors; they
            # still take part in BM25 and are simply absent from the cosine ranking.
            rows = [i for i, c in enumerate(self._chunks) if c.embedding is not None]
            dims = {int(self._chunks[i].embedding.shape[0]) for i in rows}  # type: ignore[union-attr]
            if rows and len(dims) == 1:
                mat = np.vstack([self._chunks[i].embedding for i in rows])
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._matrix = mat / norms
                self._matrix_rows = rows
            else:
                self._matrix = None
                self._matrix_rows = []
        return self._chunks

    # ---- search --------------------------------------------------------------------
    def search(self, query: str, *, k: int = SEARCH_K, url: str | None = None) -> list[SearchHit]:
        chunks = self._load()
        if not chunks:
            return []
        idxs = [i for i, c in enumerate(chunks) if url is None or c.url == url]
        if not idxs:
            return []
        rankings: list[list[int]] = []

        if self._bm25 is not None:
            q_tokens = set(tokenize(query))
            scores = self._bm25.get_scores(list(q_tokens) or ["_"])
            # Rank only chunks sharing a query term (BM25 scores can be <= 0 on tiny corpora).
            matched = [i for i in idxs if q_tokens & self._token_sets[i]]
            matched.sort(key=lambda i: float(scores[i]), reverse=True)
            rankings.append(matched[: k * 3])

        if self.embeddings is not None and self._matrix is not None:
            try:
                q = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
            except Exception as exc:
                log.warning("query embedding failed; keyword ranking only: %s", exc)
                q = np.zeros(0, dtype=np.float32)
            qn = float(np.linalg.norm(q)) if q.size == self._matrix.shape[1] else 0.0
            if qn:
                sims_rows = self._matrix @ (q / qn)
                sims = dict(zip(self._matrix_rows, sims_rows.tolist(), strict=True))
                emb = sorted((i for i in idxs if i in sims), key=lambda i: sims[i], reverse=True)
                rankings.append(emb[: k * 3])

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        hits: list[SearchHit] = []
        seen_hash: set[str] = set()
        for i, score in ordered:
            c = chunks[i]
            if c.text_hash in seen_hash:
                continue
            seen_hash.add(c.text_hash)
            hits.append(
                SearchHit(
                    url=c.url,
                    heading=c.heading,
                    excerpt=_excerpt(c.text, query),
                    ordinal=c.ordinal,
                    score=round(score, 4),
                )
            )
            if len(hits) >= k:
                break
        return hits

    def read(
        self,
        url: str,
        *,
        section: str | None = None,
        offset: int = 0,
        max_chars: int = MAX_READ_CHARS,
    ) -> tuple[str, bool, int]:
        """Return (text, more_available, next_offset) for a page or a heading section."""
        chunks = [c for c in self._load() if c.url == url]
        if section:
            needle = section.strip().lower()
            chunks = [c for c in chunks if needle in (c.heading or "").lower()] or chunks
        chunks = [c for c in chunks if c.ordinal >= offset]
        out: list[str] = []
        used = 0
        next_offset = offset
        for c in chunks:
            block = f"## {c.heading}\n{c.text}" if c.heading else c.text
            if used + len(block) > max_chars and out:
                return "\n\n".join(out), True, next_offset
            out.append(block[:max_chars])
            used += len(block)
            next_offset = c.ordinal + 1
        return "\n\n".join(out), False, next_offset

    def page_outline(self, url: str, limit: int) -> list[str]:
        seen: list[str] = []
        for c in self.store.list_chunks(url):
            if c.heading and c.heading not in seen:
                seen.append(c.heading)
            if len(seen) >= limit:
                break
        return seen


def _excerpt(text: str, query: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    """Window of the chunk around the densest query-term region (verbatim text)."""
    if len(text) <= max_chars:
        return text
    terms = set(tokenize(query))
    lower = text.lower()
    best_start, best_hits = 0, -1
    step = max(50, max_chars // 4)
    for start in range(0, max(1, len(text) - max_chars + 1), step):
        window = lower[start : start + max_chars]
        hits = sum(window.count(t) for t in terms)
        if hits > best_hits:
            best_start, best_hits = start, hits
    # Snap to word boundaries so excerpts stay verbatim substrings
    start = best_start
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    end = min(len(text), start + max_chars)
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end].strip()


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)
