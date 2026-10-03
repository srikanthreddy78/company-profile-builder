"""Chunking drops navigation noise and duplicates; hybrid search finds the right page."""

from __future__ import annotations

from profile_builder.retrieval.chunking import chunk_markdown
from profile_builder.retrieval.index import HashEmbeddings, HybridIndex
from profile_builder.state.run_store import RunStore

NAV = "\n".join(f"- [Link {i}](https://x/{i})" for i in range(12))


def test_chunking_is_heading_aware_and_drops_nav():
    md = (
        f"# Home\n\n{NAV}\n\nWe protect data in use with confidential computing enclaves for regulated industries.\n\n## Pricing\n\nPricing is per protected workload. "
        + "Contact sales for details. " * 60
    )
    chunks, _ = chunk_markdown(md, page_title="Home")
    assert all("Link 1" not in c.text for c in chunks)
    headings = {c.heading for c in chunks}
    assert "Home" in headings and "Pricing" in headings
    assert all(len(c.text) <= 1600 for c in chunks)
    assert len(chunks) >= 2


def test_cross_page_dedupe_and_hybrid_search(tmp_path):
    store = RunStore(tmp_path)
    index = HybridIndex(store, HashEmbeddings(), embedding_model="fake")
    footer = "Copyright Acme Inc. All rights reserved. Terms of service and privacy policy apply to every page."
    kept1, dropped1 = index.index_page(
        "https://a.example/product",
        f"# Product\n\nRuntime encryption protects data while it is processed in memory using hardware enclaves.\n\n{footer}",
        title="Product",
    )
    kept2, dropped2 = index.index_page(
        "https://a.example/about",
        f"# About\n\nFounded by cloud security engineers who love enclave attestation research.\n\n{footer}",
        title="About",
    )
    assert kept1 == 1 and dropped1 == 0
    assert kept2 == 1 and dropped2 == 1  # repeated footer paragraph dropped
    assert footer not in index.read("https://a.example/about")[0]
    hits = index.search("how does runtime encryption protect memory")
    assert hits and hits[0].url == "https://a.example/product"
    hits_about = index.search("who founded the company", url="https://a.example/about")
    assert hits_about and all(h.url.endswith("/about") for h in hits_about)
    text, more, _ = index.read("https://a.example/product")
    assert "Runtime encryption" in text and more is False
    assert index.page_outline("https://a.example/product", 5) == ["Product"]


def test_bm25_only_mode(tmp_path):
    store = RunStore(tmp_path)
    index = HybridIndex(store, None, use_embeddings=False)
    index.index_page(
        "https://a.example/",
        "# Home\n\nAcme sells a confidential computing platform to banks and hospitals.",
        title="Home",
    )
    assert index.search("banks hospitals")[0].url == "https://a.example/"
    assert store.usage_totals()["calls"] == 0  # no embedding usage recorded
