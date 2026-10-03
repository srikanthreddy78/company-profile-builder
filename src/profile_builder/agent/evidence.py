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
from profile_builder.schema import (
    FEATURE_FIELDS,
    FieldPathError,
    base_of,
    get_by_path,
    parse_field_path,
)
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


def unsupported_by_answer(field_path: str, value: Any, answer: str) -> list[str]:
    """The parts of `value` the user's answer does not support, checked one by one: list
    items individually, feature objects per non-empty field, a scalar as a whole. Empty
    when the answer covers everything."""
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                out.extend(unsupported_by_answer(field_path, item, answer))
            elif not supports(field_path, item, answer):
                out.append(str(item))
        return out
    if isinstance(value, dict):
        return [
            f"{key}: {val}"
            for key, val in value.items()
            if val and not supports(f"{field_path}.{key}", val, answer)
        ]
    return [] if supports(field_path, value, answer) else [str(value)]


def leaf_paths(path: str, value: Any) -> list[str]:
    """Evidence row paths for a value: list items per index, feature objects per non-empty
    field, scalars as themselves. Grounding is tracked per leaf."""
    if isinstance(value, list):
        return [p for i, item in enumerate(value) for p in leaf_paths(f"{path}[{i}]", item)]
    if isinstance(value, dict):
        return [f"{path}.{sub}" for sub in FEATURE_FIELDS if value.get(sub)]
    return [path]


def reject(path: str, code: str, reason: str) -> dict[str, Any]:
    return {"field_path": path, "reason": reason, "reason_code": code}


def descends(path: str, root: str) -> bool:
    """True when `path` is `root` itself or one of its list items / subfields."""
    return path == root or path.startswith(root + "[") or path.startswith(root + ".")


def _feature_rows(
    path: str, feat: dict[str, Any], excerpt: str, src: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows for one feature object: each non-empty subfield is checked on its own, so an
    excerpt about the mechanism never grounds an invented benefit. Returns (accepted,
    rejected); accepted is empty when the excerpt supports no subfield at all."""
    present = [sub for sub in FEATURE_FIELDS if feat.get(sub)]
    supported = [sub for sub in present if supports(f"{path}.{sub}", feat[sub], excerpt)]
    if not supported:
        return [], [
            reject(
                path,
                "NO_OVERLAP",
                "excerpt does not mention any field of this feature; quote the passage it comes from",
            )
        ]
    accepted = [
        {"field_path": f"{path}.{sub}", "source_url": src, "excerpt": excerpt} for sub in supported
    ]
    rejected = [
        reject(
            f"{path}.{sub}",
            "NO_OVERLAP",
            "excerpt does not support this feature field; cite the passage it comes from "
            "separately or leave the field empty",
        )
        for sub in present
        if sub not in supported
    ]
    return accepted, rejected


def _item_rows(
    path: str, value: Any, excerpt: str, src: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows for one list item (a string, or a feature object)."""
    if isinstance(value, dict):
        return _feature_rows(path, value, excerpt, src)
    if supports(path, value, excerpt):
        return [{"field_path": path, "source_url": src, "excerpt": excerpt}], []
    return [], [
        reject(
            path,
            "NO_OVERLAP",
            "excerpt does not mention this list item; cite the passage it comes from",
        )
    ]


def verify_website_evidence(
    ctx: ToolContext, profile: dict[str, Any], items: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Accepted rows are per leaf: evidence cited for a whole list is recorded per item
    (`base[i]`) for exactly the items the excerpt supports, and evidence cited for a feature
    object is recorded per subfield (`base[i].how_it_works`) for exactly the subfields it
    supports; unsupported items/subfields are listed back so the model can cite them
    separately."""
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
            acc_rows: list[dict[str, Any]] = []
            rej_rows: list[dict[str, Any]] = []
            for i, item in enumerate(value):
                acc, rej = _item_rows(f"{path}[{i}]", item, excerpt, src_norm)
                acc_rows.extend(acc)
                rej_rows.extend(rej)
            if not acc_rows:
                rejected.append(
                    reject(
                        path,
                        "NO_OVERLAP",
                        "excerpt does not mention any item of this list; cite one passage per item",
                    )
                )
                continue
            accepted.extend(acc_rows)
            rejected.extend(rej_rows)
            continue
        if isinstance(value, dict):  # one feature object: grounded subfield by subfield
            acc, rej = _feature_rows(path, value, excerpt, src_norm)
            accepted.extend(acc)
            rejected.extend(rej)
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
