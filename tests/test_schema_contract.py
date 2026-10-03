"""The exported JSON must match the assignment template exactly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from profile_builder.schema import (
    TEMPLATE,
    CompanyBrain,
    FieldPathError,
    get_by_path,
    json_schema,
    parse_field_path,
    set_by_path,
)


def _keys(d, prefix=""):
    out = set()
    for k, v in d.items():
        path = f"{prefix}{k}"
        out.add(path)
        if isinstance(v, dict):
            out |= _keys(v, path + ".")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out |= _keys(v[0], path + "[].")
    return out


def test_empty_profile_matches_template_keys_and_types():
    empty = CompanyBrain().model_dump(mode="json")
    template_keys = _keys(TEMPLATE) - _keys(
        TEMPLATE["product"]["features_and_capabilities"][0], "product.features_and_capabilities[]."
    )
    assert _keys(empty) == template_keys
    assert empty["artifact"] == "company_brain" and empty["version"] == 1
    assert empty["product"]["features_and_capabilities"] == []  # no placeholder objects


def test_feature_item_shape_matches_template():
    feat = {"name": "x", "description": "", "how_it_works": "", "customer_benefit": ""}
    brain = CompanyBrain.model_validate(
        {**TEMPLATE, "product": {**TEMPLATE["product"], "features_and_capabilities": [feat]}}
    )
    assert brain.product.features_and_capabilities[0].model_dump() == feat


@pytest.mark.parametrize(
    "bad",
    [
        {"company": {"name": "unknown", "website_url": ""}},
        {"company": {"name": None, "website_url": "", "extra": 1}},
        {"customer": {"target_customer": "N/A"}},
        {
            "product": {
                "features_and_capabilities": [
                    {"name": "", "description": "", "how_it_works": "", "customer_benefit": ""}
                ]
            }
        },
        {"artifact": "something_else"},
        {"version": 2},
        {"customer": {"buyers": "CISOs"}},
    ],
)
def test_filler_and_shape_violations_rejected(bad):
    with pytest.raises(ValidationError):
        CompanyBrain.model_validate({**CompanyBrain().model_dump(), **bad})


def test_list_filler_items_are_dropped_not_fatal():
    brain = CompanyBrain.model_validate(
        {
            **CompanyBrain().model_dump(),
            "customer": {**CompanyBrain().customer.model_dump(), "buyers": ["CISO", "unknown", ""]},
        }
    )
    assert brain.customer.buyers == ["CISO"]


def test_json_schema_file_in_sync():
    path = Path(__file__).resolve().parents[1] / "schema" / "company_brain.schema.json"
    assert path.exists(), "run `make schema`"
    assert json.loads(path.read_text()) == json_schema()


def test_field_paths():
    assert parse_field_path("customer.buyers[2]") == ("customer.buyers", 2, None)
    assert parse_field_path("product.features_and_capabilities[0].how_it_works") == (
        "product.features_and_capabilities",
        0,
        "how_it_works",
    )
    for bad in [
        "company.name[0]",
        "product.features_and_capabilities.description",
        "nope.field",
        "customer.buyers.x",
        "product.features_and_capabilities[0].bogus",
    ]:
        with pytest.raises(FieldPathError):
            parse_field_path(bad)
    data = CompanyBrain().model_dump()
    set_by_path(data, "customer.buyers[0]", "CISO")
    set_by_path(data, "product.features_and_capabilities[0].name", "Enclaves")
    set_by_path(data, "product.features_and_capabilities[0].how_it_works", "SGX")
    assert get_by_path(data, "customer.buyers") == ["CISO"]
    assert get_by_path(data, "product.features_and_capabilities[0].how_it_works") == "SGX"
    with pytest.raises(FieldPathError):
        set_by_path(data, "customer.buyers[5]", "gap")


def test_strip_unknown_keys():
    from profile_builder.schema import strip_unknown_keys

    data = {
        **TEMPLATE,
        "content_evidence": {
            **TEMPLATE["content_evidence"],
            "proprietary_insights_or_examples_additional": ["x"],
        },
        "extra_top": 1,
    }
    data["product"] = {
        **TEMPLATE["product"],
        "features_and_capabilities": [
            {
                "name": "A",
                "description": "",
                "how_it_works": "",
                "customer_benefit": "",
                "source": "u",
            }
        ],
    }
    cleaned, dropped = strip_unknown_keys(data)
    assert set(dropped) == {
        "content_evidence.proprietary_insights_or_examples_additional",
        "extra_top",
        "product.features_and_capabilities[0].source",
    }
    CompanyBrain.model_validate(cleaned)
