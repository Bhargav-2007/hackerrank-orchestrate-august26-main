"""Compatibility wrapper for older imports."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def get_evidence_and_confidence(
    original_message: Dict[str, Any],
    enriched_message: Dict[str, Any],
    features: Dict[str, Any],
    action: str,
    rule_confidence: float,
    message_history: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], float]:
    return ["none"], max(0.1, min(0.95, rule_confidence))