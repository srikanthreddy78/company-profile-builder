"""The company_brain.json output contract.

Mirrors the assignment template exactly: same keys, nesting and value types. No metadata,
no Fact wrappers. Unknown strings are ``""`` and unknown collections are ``[]``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FILLER_VALUES = frozenset(
    {"unknown", "n/a", "na", "none", "null", "tbd", "todo", "not available", "not specified", "-"}
)


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"expected a string, got {type(value).__name__}")
    stripped = value.strip()
    # Values are statements, not quotations: drop wrapping quote marks the model may copy over.
    while len(stripped) >= 2 and stripped[0] in "\"“”'‘’" and stripped[-1] in "\"“”'‘’":
        stripped = stripped[1:-1].strip()
    if stripped.lower() in FILLER_VALUES:
        raise ValueError(f'filler value {value!r} is not allowed; use "" for unknown')
    return stripped


def _clean_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"expected a list of strings, got {type(values).__name__}")
    out: list[str] = []
    for v in values:
        try:
            s = _clean_str(v)
        except ValueError:
            continue  # filler items ("unknown", "N/A") are dropped, not fatal
        if s:
            out.append(s)
    return out


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Company(_Strict):
    name: str = ""
    website_url: str = ""

    _clean = field_validator("name", "website_url", mode="before")(_clean_str)


class FeatureCapability(_Strict):
    name: str = ""
    description: str = ""
    how_it_works: str = ""
    customer_benefit: str = ""

    _clean = field_validator(
        "name", "description", "how_it_works", "customer_benefit", mode="before"
    )(_clean_str)

    @model_validator(mode="after")
    def _not_placeholder(self) -> FeatureCapability:
        if not any([self.name, self.description, self.how_it_works, self.customer_benefit]):
            raise ValueError("placeholder feature object (all fields empty) is not allowed")
        return self


class Product(_Strict):
    name: str = ""
    description: str = ""
    positioning: str = ""
    features_and_capabilities: list[FeatureCapability] = Field(default_factory=list)
    differentiators: list[str] = Field(default_factory=list)

    _clean = field_validator("name", "description", "positioning", mode="before")(_clean_str)
    _clean_list = field_validator("differentiators", mode="before")(_clean_str_list)


class Customer(_Strict):
    target_customer: str = ""
    buyers: list[str] = Field(default_factory=list)
    users: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    desired_outcomes: list[str] = Field(default_factory=list)
    existing_alternatives: list[str] = Field(default_factory=list)

    _clean = field_validator("target_customer", mode="before")(_clean_str)
    _clean_list = field_validator(
        "buyers",
        "users",
        "problems",
        "use_cases",
        "desired_outcomes",
        "existing_alternatives",
        mode="before",
    )(_clean_str_list)


class ContentEvidence(_Strict):
    customer_stories: list[str] = Field(default_factory=list)
    company_expertise: list[str] = Field(default_factory=list)
    product_evidence: list[str] = Field(default_factory=list)
    proprietary_insights_or_examples: list[str] = Field(default_factory=list)

    _clean_list = field_validator(
        "customer_stories",
        "company_expertise",
        "product_evidence",
        "proprietary_insights_or_examples",
        mode="before",
    )(_clean_str_list)


class Brand(_Strict):
    voice_and_tone: list[str] = Field(default_factory=list)
    writing_style: list[str] = Field(default_factory=list)
    preferred_terms: list[str] = Field(default_factory=list)
    terms_or_claims_to_avoid: list[str] = Field(default_factory=list)

    _clean_list = field_validator(
        "voice_and_tone",
        "writing_style",
        "preferred_terms",
        "terms_or_claims_to_avoid",
        mode="before",
    )(_clean_str_list)


class CompanyBrain(_Strict):
    artifact: Literal["company_brain"] = "company_brain"
    version: Literal[1] = 1
    company: Company = Field(default_factory=Company)
    product: Product = Field(default_factory=Product)
    customer: Customer = Field(default_factory=Customer)
    content_evidence: ContentEvidence = Field(default_factory=ContentEvidence)
    brand: Brand = Field(default_factory=Brand)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"

    @classmethod
    def empty(cls) -> CompanyBrain:
        return cls()


# --------------------------------------------------------------------------------------
# Field paths (used by evidence, gaps, and the update tool)
# --------------------------------------------------------------------------------------

STRING_PATHS: tuple[str, ...] = (
    "company.name",
    "company.website_url",
    "product.name",
    "product.description",
    "product.positioning",
    "customer.target_customer",
)
STRING_LIST_PATHS: tuple[str, ...] = (
    "product.differentiators",
    "customer.buyers",
    "customer.users",
    "customer.problems",
    "customer.use_cases",
    "customer.desired_outcomes",
    "customer.existing_alternatives",
    "content_evidence.customer_stories",
    "content_evidence.company_expertise",
    "content_evidence.product_evidence",
    "content_evidence.proprietary_insights_or_examples",
    "brand.voice_and_tone",
    "brand.writing_style",
    "brand.preferred_terms",
    "brand.terms_or_claims_to_avoid",
)
FEATURE_LIST_PATH = "product.features_and_capabilities"
FEATURE_FIELDS: tuple[str, ...] = ("name", "description", "how_it_works", "customer_benefit")
# Every top-level field a draft can carry over from an earlier one (strings, lists, features).
MERGEABLE_BASES: tuple[str, ...] = (*STRING_PATHS, *STRING_LIST_PATHS, FEATURE_LIST_PATH)

_PATH_RE = re.compile(r"^(?P<base>[a-z_]+\.[a-z_]+)(?:\[(?P<idx>\d+)\])?(?:\.(?P<sub>[a-z_]+))?$")


class FieldPathError(ValueError):
    pass


def base_of(path: str) -> str:
    """The top-level field of a path: ``customer.buyers[2]`` → ``customer.buyers``."""
    return path.split("[")[0]


def parse_field_path(path: str) -> tuple[str, int | None, str | None]:
    """Validate a field path against the contract and return (base, index, subfield)."""
    m = _PATH_RE.match(path.strip())
    if not m:
        raise FieldPathError(f"malformed field path {path!r}")
    base, idx, sub = m.group("base"), m.group("idx"), m.group("sub")
    index = int(idx) if idx is not None else None
    if base in STRING_PATHS:
        if index is not None or sub:
            raise FieldPathError(f"{base} is a string; indexes/subfields are not allowed")
        return base, None, None
    if base in STRING_LIST_PATHS:
        if sub:
            raise FieldPathError(f"{base} is a list of strings; subfields are not allowed")
        return base, index, None
    if base == FEATURE_LIST_PATH:
        if sub is not None and sub not in FEATURE_FIELDS:
            raise FieldPathError(f"unknown feature field {sub!r}")
        if sub is not None and index is None:
            raise FieldPathError("feature subfields require an index, e.g. [0].description")
        return base, index, sub
    raise FieldPathError(f"unknown field path {path!r}")


def _get_container(data: dict[str, Any], base: str) -> tuple[dict[str, Any], str]:
    section, key = base.split(".", 1)
    return data[section], key


def get_by_path(data: dict[str, Any], path: str) -> Any:
    base, index, sub = parse_field_path(path)
    container, key = _get_container(data, base)
    value = container[key]
    if index is not None:
        if index >= len(value):
            return None
        value = value[index]
    if sub is not None:
        value = value[sub] if isinstance(value, dict) else None
    return value


def set_by_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Set a value in a plain dict representation of CompanyBrain (validated afterwards)."""
    base, index, sub = parse_field_path(path)
    container, key = _get_container(data, base)
    if index is None and sub is None:
        container[key] = value
        return
    lst = container[key]
    if not isinstance(lst, list):
        raise FieldPathError(f"{base} is not a list")
    # index == len(lst) appends (the "[len]" convention of apply_profile_updates); anything past
    # that is an error so a typo cannot create holes.
    if index is not None and index > len(lst):
        raise FieldPathError(f"index {index} out of range for {base} (len={len(lst)})")
    if sub is None:
        if index == len(lst):
            lst.append(value)
        else:
            lst[index] = value
        return
    if index == len(lst):
        lst.append({f: "" for f in FEATURE_FIELDS})
    item = lst[index]
    if not isinstance(item, dict):
        raise FieldPathError(f"{base}[{index}] is not a feature object")
    item[sub] = value


def iter_leaf_paths(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """Every leaf field path with its value, including list items and feature subfields."""
    leaves: list[tuple[str, Any]] = []
    for p in STRING_PATHS:
        leaves.append((p, get_by_path(data, p)))
    for p in STRING_LIST_PATHS:
        items = get_by_path(data, p) or []
        if not items:
            leaves.append((p, []))
        for i, item in enumerate(items):
            leaves.append((f"{p}[{i}]", item))
    features = get_by_path(data, FEATURE_LIST_PATH) or []
    if not features:
        leaves.append((FEATURE_LIST_PATH, []))
    for i, feat in enumerate(features):
        for f in FEATURE_FIELDS:
            leaves.append((f"{FEATURE_LIST_PATH}[{i}].{f}", feat.get(f, "")))
    return leaves


def ancestors(path: str) -> list[str]:
    """Paths whose evidence also covers `path` (list → item, feature → subfield)."""
    base, index, sub = parse_field_path(path)
    out = [path]
    if sub is not None:
        out.append(f"{base}[{index}]")
    if index is not None:
        out.append(base)
    return out


SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "company": tuple(Company.model_fields),
    "product": tuple(Product.model_fields),
    "customer": tuple(Customer.model_fields),
    "content_evidence": tuple(ContentEvidence.model_fields),
    "brand": tuple(Brand.model_fields),
}


def strip_unknown_keys(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop keys that are not part of the contract (the model occasionally invents
    `..._additional` fields). Returns the cleaned copy and the dropped key paths."""
    dropped: list[str] = []
    out: dict[str, Any] = {}
    for key, value in (data or {}).items():
        if key in ("artifact", "version"):
            out[key] = value
        elif key in SECTION_KEYS and isinstance(value, dict):
            section: dict[str, Any] = {}
            for sub, sub_val in value.items():
                if sub in SECTION_KEYS[key]:
                    section[sub] = sub_val
                else:
                    dropped.append(f"{key}.{sub}")
            if key == "product" and isinstance(section.get("features_and_capabilities"), list):
                feats = []
                for i, feat in enumerate(section["features_and_capabilities"]):
                    if isinstance(feat, dict):
                        keep = {k: v for k, v in feat.items() if k in FEATURE_FIELDS}
                        dropped.extend(
                            f"product.features_and_capabilities[{i}].{k}"
                            for k in feat
                            if k not in FEATURE_FIELDS
                        )
                        feats.append(keep)
                    else:
                        feats.append(feat)
                section["features_and_capabilities"] = feats
            out[key] = section
        elif key in SECTION_KEYS:
            out[key] = value  # wrong type: let validation report it
        else:
            dropped.append(key)
    return out, dropped


def _require_all_properties(node: Any) -> None:
    """Every object schema lists all of its properties as required: the contract always
    emits every key (unknowns are "" / []), so consumers can rely on their presence."""
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" and isinstance(node.get("properties"), dict):
        node["required"] = list(node["properties"])
    for key in ("properties", "$defs"):
        for child in (node.get(key) or {}).values():
            _require_all_properties(child)
    for key in ("items", "additionalProperties"):
        if isinstance(node.get(key), dict):
            _require_all_properties(node[key])
    for key in ("anyOf", "oneOf", "allOf"):
        for child in node.get(key) or []:
            _require_all_properties(child)


def json_schema() -> dict[str, Any]:
    schema = CompanyBrain.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "company_brain"
    _require_all_properties(schema)
    return schema


def write_json_schema(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(), indent=2) + "\n", encoding="utf-8")
    return path


TEMPLATE: dict[str, Any] = {
    "artifact": "company_brain",
    "version": 1,
    "company": {"name": "", "website_url": ""},
    "product": {
        "name": "",
        "description": "",
        "positioning": "",
        "features_and_capabilities": [
            {"name": "", "description": "", "how_it_works": "", "customer_benefit": ""}
        ],
        "differentiators": [],
    },
    "customer": {
        "target_customer": "",
        "buyers": [],
        "users": [],
        "problems": [],
        "use_cases": [],
        "desired_outcomes": [],
        "existing_alternatives": [],
    },
    "content_evidence": {
        "customer_stories": [],
        "company_expertise": [],
        "product_evidence": [],
        "proprietary_insights_or_examples": [],
    },
    "brand": {
        "voice_and_tone": [],
        "writing_style": [],
        "preferred_terms": [],
        "terms_or_claims_to_avoid": [],
    },
}
