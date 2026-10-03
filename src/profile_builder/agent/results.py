"""Tool results are compact JSON strings the model reads back. These helpers keep the
envelope uniform: ``{"ok": true, ...}`` on success, ``{"ok": false, "error": ...}`` on a
rejected call."""

from __future__ import annotations

import json
from typing import Any


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=None, default=str)


def ok(**fields: Any) -> str:
    return to_json({"ok": True, **fields})


def fail(error: str, *, code: str | None = None, **fields: Any) -> str:
    payload: dict[str, Any] = {"ok": False, "error": error}
    if code:
        payload["error_code"] = code
    payload.update(fields)
    return to_json(payload)
