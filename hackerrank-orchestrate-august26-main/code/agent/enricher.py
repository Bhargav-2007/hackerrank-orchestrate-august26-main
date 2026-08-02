"""Compatibility wrapper for older imports."""

from __future__ import annotations

from typing import Any, Dict


def enrich_message(message: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    enriched = dict(message)
    enriched.update(kwargs.get("extra", {}))
    return enriched