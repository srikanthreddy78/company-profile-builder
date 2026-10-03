"""Evidence checks: does an excerpt or user answer actually support a field value, and is
a cited website excerpt a verbatim quote from a page fetched in this run.

Rejections carry a stable ``reason_code`` (BAD_PATH, FIELD_EMPTY, SOURCE_NOT_FETCHED,
TOO_SHORT, TOO_LONG, NOT_VERBATIM, NO_OVERLAP, ...) so the model can act on them."""

from __future__ import annotations

from typing import Any

from profile_builder.agent.context import ToolContext
from profile_builder.config import (
    MAX_EVIDENCE_EXCERPT_CHARS,
    MIN_CONTENT_TOKEN_CHARS,
    MIN_EXCERPT_CHARS,
    OBSERVATION_PATHS,
    STEM_CHARS,
)
from profile_builder.retrieval.index import tokenize
from profile_builder.schema import FieldPathError, base_of, get_by_path, parse_field_path
from profile_builder.security import excerpt_in_page
from profile_builder.state.run_store import RunStore
from profile_builder.web.url_guard import normalize_url
from profile_builder.workflow.export import evidence_paths
from profile_builder.workflow.gaps import grounding_report


def content_stems(text: str) -> set[str]:
    """Content words reduced to a short stem so inflections (key/keys, release/releases,
    protect/protection) still count as overlap."""
    return {t[:STEM_CHARS] for t in tokenize(text) if len(t) >= MIN_CONTENT_TOKEN_CHARS}


def value_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(value_text(v) for v in value)
    return str(value)


def supports(field_path: str, value: Any, text: str) -> bool:
    """An excerpt/answer supports a value if they share a content-word stem. Observed brand
    patterns (tone, writing style) are exempt: their evidence is an illustrative passage."""
    if base_of(field_path) in OBSERVATION_PATHS:
        return True
    vt = content_stems(value_text(value))
    return not vt or bool(vt & content_stems(text))


def reject(path: str, code: str, reason: str) -> dict[str, Any]:
    return {"field_path": path, "reason": reason, "reason_code": code}


def descends(path: str, root: str) -> bool:
    """True when `path` is `root` itself or one of its list items / subfields."""
    return path == root or path.startswith(root + "[") or path.startswith(root + ".")


def verify_website_evidence(
    ctx: ToolContext, profile: dict[str, Any], items: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Accepted rows are per leaf: evidence cited for a whole list is recorded per item
    (`base[i]`) for exactly the items the excerpt supports; unsupported items are listed
    back so the model can cite them separately."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    fetched = ctx.store.fetched_urls()
    for item in items or []:
        if not isinstance(item, dict):
            rejected.append(reject("", "BAD_PATH", "evidence item must be an object"))
            continue
        path = str(item.get("field_path", "")).strip()
        src = str(item.get("source_url", "")).strip()
        excerpt = str(item.get("excerpt", "")).strip()
        try:
            _base, index, _sub = parse_field_path(path)
        except FieldPathError as exc:
            rejected.append(reject(path, "BAD_PATH", str(exc)))
            continue
        value = get_by_path(profile, path)
        if value in ("", [], None):
            rejected.append(reject(path, "FIELD_EMPTY", "field is empty in the profile"))
            continue
        try:
            src_norm = normalize_url(src) if src else ""
        except ValueError:
            src_norm = ""
        if src_norm not in fetched:
            rejected.append(
                reject(path, "SOURCE_NOT_FETCHED", "source_url is not a page fetched in this run")
            )
            continue
        if len(excerpt) < MIN_EXCERPT_CHARS:
            rejected.append(
                reject(
                    path,
                    "TOO_SHORT",
                    f"excerpt too short to ground a claim (min {MIN_EXCERPT_CHARS} chars)",
                )
            )
            continue
        if len(excerpt) > MAX_EVIDENCE_EXCERPT_CHARS:
            rejected.append(
                reject(
                    path,
                    "TOO_LONG",
                    f"excerpt too long (max {MAX_EVIDENCE_EXCERPT_CHARS} chars); quote only the relevant passage",
                )
            )
            continue
        page_text = ctx.page_text(src_norm) or ""
        if not excerpt_in_page(excerpt, page_text):
            rejected.append(
                reject(path, "NOT_VERBATIM", "excerpt is not a verbatim quote from that page")
            )
            continue
        if index is None and isinstance(value, list):
            supported = [i for i, v in enumerate(value) if supports(path, v, excerpt)]
            if not supported:
                rejected.append(
                    reject(
                        path,
                        "NO_OVERLAP",
                        "excerpt does not mention any item of this list; cite one passage per item",
                    )
                )
                continue
            for i in range(len(value)):
                if i in supported:
                    accepted.append(
                        {"field_path": f"{path}[{i}]", "source_url": src_norm, "excerpt": excerpt}
                    )
                else:
                    rejected.append(
                        reject(
                            f"{path}[{i}]",
                            "NO_OVERLAP",
                            "excerpt does not mention this list item; cite the passage it comes from",
                        )
                    )
            continue
        if not supports(path, value, excerpt):
            rejected.append(
                reject(
                    path,
                    "NO_OVERLAP",
                    "excerpt does not mention the field value; quote the passage the value comes from",
                )
            )
            continue
        accepted.append({"field_path": path, "source_url": src_norm, "excerpt": excerpt})
    return accepted, rejected


def ungrounded_paths(store: RunStore, profile: dict[str, Any]) -> list[str]:
    return list(grounding_report(profile, evidence_paths(store))["ungrounded"])
