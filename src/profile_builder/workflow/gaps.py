"""Gap analysis: which contract fields are empty, which matter most, and which are
disputed or ungrounded. Pure functions over the draft dict + run state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profile_builder.schema import (
    FEATURE_LIST_PATH,
    STRING_LIST_PATHS,
    STRING_PATHS,
    ancestors,
    get_by_path,
    iter_leaf_paths,
)

# Higher = more important to resolve via interview. Fields affecting product scope,
# target customer, differentiation and the business goal come first (per the brief).
PRIORITY: dict[str, int] = {
    "product.name": 100,
    "product.description": 95,
    "customer.target_customer": 95,
    "product.positioning": 90,
    "product.differentiators": 90,
    "customer.problems": 80,
    "customer.use_cases": 80,
    "customer.buyers": 75,
    "customer.users": 75,
    "customer.desired_outcomes": 70,
    "customer.existing_alternatives": 65,
    "product.features_and_capabilities": 85,
    "content_evidence.customer_stories": 50,
    "content_evidence.product_evidence": 50,
    "content_evidence.company_expertise": 45,
    "content_evidence.proprietary_insights_or_examples": 35,
    "brand.voice_and_tone": 40,
    "brand.writing_style": 30,
    "brand.preferred_terms": 35,
    "brand.terms_or_claims_to_avoid": 40,
    "company.name": 60,
    "company.website_url": 20,
}

REASONS: dict[str, str] = {
    "product.name": "which single product this profile covers",
    "product.description": "what the product actually does",
    "product.positioning": "how the company positions the product against the market",
    "product.differentiators": "why customers pick it over alternatives",
    "customer.target_customer": "which customer segment to prioritize",
    "customer.problems": "the customer pains the product solves",
    "customer.use_cases": "the concrete tasks or situations it is used for",
    "customer.buyers": "who makes the purchasing decision",
    "customer.users": "who uses the product day to day",
    "customer.desired_outcomes": "the results customers want from the product",
    "customer.existing_alternatives": "competing approaches or tools customers use today",
    "product.features_and_capabilities": "the supported capabilities of the product",
    "brand.terms_or_claims_to_avoid": "claims or terms the company does not want used",
    "brand.preferred_terms": "the company's preferred terminology",
    "brand.voice_and_tone": "the tone the content should use",
}


@dataclass
class Gap:
    field_path: str
    priority: int
    reason: str
    kind: str  # empty | partial_feature | ungrounded | conflict

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_path": self.field_path,
            "priority": self.priority,
            "reason": self.reason,
            "kind": self.kind,
        }


def empty_gaps(profile: dict[str, Any]) -> list[Gap]:
    gaps: list[Gap] = []
    for path in STRING_PATHS:
        if not get_by_path(profile, path):
            gaps.append(Gap(path, PRIORITY.get(path, 10), REASONS.get(path, "missing"), "empty"))
    for path in STRING_LIST_PATHS:
        if not get_by_path(profile, path):
            gaps.append(Gap(path, PRIORITY.get(path, 10), REASONS.get(path, "missing"), "empty"))
    features = get_by_path(profile, FEATURE_LIST_PATH) or []
    if not features:
        gaps.append(
            Gap(FEATURE_LIST_PATH, PRIORITY[FEATURE_LIST_PATH], REASONS[FEATURE_LIST_PATH], "empty")
        )
    for i, feat in enumerate(features):
        for sub in ("description", "how_it_works", "customer_benefit"):
            if not feat.get(sub):
                gaps.append(
                    Gap(
                        f"{FEATURE_LIST_PATH}[{i}].{sub}",
                        PRIORITY[FEATURE_LIST_PATH] - 30 - (10 if sub == "how_it_works" else 0),
                        f"{sub.replace('_', ' ')} of capability '{feat.get('name') or i}' is unknown",
                        "partial_feature",
                    )
                )
    gaps.sort(key=lambda g: g.priority, reverse=True)
    return gaps


def grounding_report(profile: dict[str, Any], evidence_paths: set[str]) -> dict[str, Any]:
    """Count populated leaves and how many have evidence (own path or an ancestor path)."""
    populated = 0
    grounded = 0
    ungrounded: list[str] = []
    for path, value in iter_leaf_paths(profile):
        if value in ("", [], None):
            continue
        populated += 1
        if any(a in evidence_paths for a in ancestors(path)):
            grounded += 1
        else:
            ungrounded.append(path)
    return {"populated": populated, "grounded": grounded, "ungrounded": ungrounded}


def section_coverage(profile: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Per top-level section: (filled leaf fields, total leaf fields considered)."""
    out: dict[str, list[int]] = {}
    for path in (*STRING_PATHS, *STRING_LIST_PATHS, FEATURE_LIST_PATH):
        section = path.split(".", 1)[0]
        filled, total = out.setdefault(section, [0, 0])
        total += 1
        if get_by_path(profile, path):
            filled += 1
        out[section] = [filled, total]
    return {k: (v[0], v[1]) for k, v in out.items()}


def prioritize_for_interview(
    profile: dict[str, Any],
    *,
    open_conflicts: list[dict[str, Any]],
    asked_paths: set[str],
    ungrounded: list[str],
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Ordered list the agent sees after each draft: conflicts first, then empty fields,
    skipping anything already asked about."""
    items: list[Gap] = []
    for c in open_conflicts:
        if c["field_path"] in asked_paths:
            continue
        items.append(
            Gap(c["field_path"], 110, f"conflicting evidence: {c.get('summary') or ''}", "conflict")
        )
    for g in empty_gaps(profile):
        if g.field_path in asked_paths or g.field_path.split("[")[0] in asked_paths:
            continue
        items.append(g)
    for p in ungrounded[:5]:
        items.append(
            Gap(p, 15, "populated without verified evidence; confirm or cite", "ungrounded")
        )
    items.sort(key=lambda g: g.priority, reverse=True)
    return [g.to_dict() for g in items[:limit]]
