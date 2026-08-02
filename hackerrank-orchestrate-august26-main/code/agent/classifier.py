"""Compatibility wrapper for older imports."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def classify_message(features: Dict[str, Any]) -> Tuple[str, float]:
    text = (features.get("text") or "").lower()
    if any(token in text for token in ["otp", "password", "blocked", "verify"]):
        return "mute", 0.9
    if any(token in text for token in ["urgent", "asap", "deadline", "call"]):
        return "notify", 0.7
    if any(token in text for token in ["sale", "offer", "discount", "reminder", "form"]):
        return "digest", 0.6
    return "digest", 0.5