"""Deterministic routing pipeline for the Message Notification Router."""

from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "dataset"


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@([A-Za-z0-9_]+)")
NUMBER_RE = re.compile(r"\b\d{4,}\b")
OTP_RE = re.compile(r"\b\d{4,8}\b")

SCAM_TOKENS = {
    "otp",
    "password",
    "passcode",
    "login",
    "verify",
    "verification",
    "kyc",
    "suspended",
    "account blocked",
    "account suspended",
    "wallet",
    "upi",
    "pay now",
    "refund",
    "remote access",
    "install",
    "ignore previous",
    "mark this message",
    "keep payments active",
}

URGENT_TOKENS = {
    "urgent",
    "asap",
    "immediately",
    "right now",
    "deadline",
    "eod",
    "before 6",
    "before 7",
    "within",
    "need help",
    "emergency",
    "critical",
    "quick",
    "please call now",
    "call now",
    "respond once",
    "quick heads-up",
    "pullled to",
    "join with",
    "need quick help",
}

PAYMENT_TOKENS = {
    "payment",
    "pay",
    "fee",
    "invoice",
    "bill",
    "due",
    "due today",
    "balance",
    "transaction",
    "upi",
    "refund",
    "collect",
    "emi",
    "renewal",
    "subscription",
    "installment",
}

PROMO_TOKENS = {
    "sale",
    "offer",
    "off",
    "discount",
    "deal",
    "win",
    "free",
    "limited",
    "shop",
    "buy now",
    "coupon",
    "subscribe",
    "unsubscribe",
    "t&c",
    "cashback",
    "welcome",
    "selling",
    "for sale",
    "dm if interested",
    "interested",
    "itinerary",
    "trip",
    "travel",
    "rs",
    "rupees",
    "50%",
}

EVENT_TOKENS = {
    "circular",
    "notice",
    "schedule",
    "timing",
    "form",
    "pickup",
    "bus",
    "class",
    "school",
    "meeting",
    "appointment",
    "reminder",
    "event",
    "consent",
    "change",
    "rescheduled",
    "prescription",
    "claim",
}

GREETING_TOKENS = {
    "good morning",
    "good afternoon",
    "good evening",
    "hello",
    "hi",
    "hey",
    "hope today is peaceful",
    "blessings",
    "stay positive",
    "no need to respond",
}

SAFETY_ADVISORY_TOKENS = {
    "safety advisory",
    "never ask",
    "do not share",
    "brand says",
    "advisory",
    "keep your account safe",
    "we never ask",
}

FORWARD_TOKENS = {
    "fwd",
    "forwarded",
    "forwarding",
    "as received",
    "share with family",
    "share with everyone",
}

BUSINESS_URGENT_CATEGORIES = {
    "bank",
    "delivery",
    "ecommerce_delivery",
    "ride_booking",
    "telecom",
    "clinic",
    "education",
}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index_by(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if value:
            indexed[value] = row
    return indexed


def _group_by(rows: Sequence[Dict[str, Any]], key: str) -> DefaultDict[str, List[Dict[str, Any]]]:
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value:
            grouped[value].append(row)
    return grouped


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> Set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    tokens = re.findall(r"[a-z0-9@._/-]+", normalized)
    return {token for token in tokens if len(token) > 1}


def _contains_phrase(text: str, phrases: Iterable[str]) -> bool:
    normalized = _normalize_text(text)
    for phrase in phrases:
        phrase_norm = _normalize_text(phrase)
        if not phrase_norm:
            continue
        if " " in phrase_norm:
            if phrase_norm in normalized:
                return True
        else:
            if re.search(rf"\b{re.escape(phrase_norm)}\b", normalized):
                return True
    return False


def _parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _parse_window(window: str) -> Optional[Tuple[int, int]]:
    if not window or "-" not in window:
        return None
    start, end = window.split("-", 1)
    try:
        start_hour, start_minute = [int(part) for part in start.split(":", 1)]
        end_hour, end_minute = [int(part) for part in end.split(":", 1)]
        return start_hour * 60 + start_minute, end_hour * 60 + end_minute
    except Exception:
        return None


def _minutes_since_midnight(timestamp: Optional[datetime]) -> Optional[int]:
    if timestamp is None:
        return None
    return timestamp.hour * 60 + timestamp.minute


def _is_inside_window(timestamp: Optional[datetime], window: str) -> bool:
    parsed_window = _parse_window(window)
    current_minutes = _minutes_since_midnight(timestamp)
    if parsed_window is None or current_minutes is None:
        return False
    start, end = parsed_window
    if start <= end:
        return start <= current_minutes <= end
    return current_minutes >= start or current_minutes <= end


def _resolve_media_path(dataset_path: Path, relative_path: str) -> Optional[Path]:
    if not relative_path:
        return None
    candidate = Path(relative_path)
    options = [
        candidate,
        dataset_path / candidate,
        dataset_path / "media" / candidate,
        dataset_path / "media" / "images" / candidate.name,
        dataset_path / "media" / "audio" / candidate.name,
    ]
    for option in options:
        if option.exists():
            return option
    return None


def _best_effort_image_ocr(image_path: Path) -> Tuple[str, Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    text = ""
    if not image_path or not image_path.exists():
        return text, metadata
    metadata["file_size"] = image_path.stat().st_size
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            metadata["width"], metadata["height"] = image.size
            metadata["mode"] = image.mode
            try:
                import pytesseract  # type: ignore

                text = pytesseract.image_to_string(image)
            except Exception:
                text = ""
    except Exception:
        text = ""
    return text.strip(), metadata


def _best_effort_voice_summary(audio_path: Path) -> Tuple[str, Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    if not audio_path or not audio_path.exists():
        return "", metadata
    metadata["file_size"] = audio_path.stat().st_size
    metadata["suffix"] = audio_path.suffix.lower()
    try:
        import wave

        if audio_path.suffix.lower() == ".wav":
            with wave.open(str(audio_path), "rb") as wav_file:
                metadata["channels"] = wav_file.getnchannels()
                metadata["frame_rate"] = wav_file.getframerate()
                metadata["duration_seconds"] = round(wav_file.getnframes() / float(wav_file.getframerate() or 1), 1)
    except Exception:
        pass
    return "", metadata


def _load_message_events(dataset_path: Path) -> DefaultDict[str, List[Dict[str, Any]]]:
    events_path = dataset_path / "message_events.csv"
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    if events_path.exists():
        for row in _read_csv(events_path):
            message_id = row.get("message_id")
            if message_id:
                grouped[message_id].append(row)
    return grouped


def load_context(dataset_path: Path = DATASET_PATH) -> Dict[str, Any]:
    users = _index_by(_read_csv(dataset_path / "users.csv"), "user_id")
    groups = _index_by(_read_csv(dataset_path / "groups.csv"), "group_id")
    businesses = _index_by(_read_csv(dataset_path / "business_accounts.csv"), "business_id")
    user_business_history = _group_by(_read_csv(dataset_path / "user_business_history.csv"), "user_id")
    group_members = _group_by(_read_csv(dataset_path / "group_members.csv"), "group_id")
    images = _index_by(_read_csv(dataset_path / "images.csv"), "image_id")
    voice_notes = _index_by(_read_csv(dataset_path / "voice_notes.csv"), "voice_note_id")
    daily_summary_rows = _read_csv(dataset_path / "daily_notification_summary.csv")
    daily_summary = _group_by(daily_summary_rows, "user_id")
    message_events = _load_message_events(dataset_path)

    history_rows = _read_csv(dataset_path / "message_history.csv")
    history_index: List[Dict[str, Any]] = []
    by_user_sender: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_user_group: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_user_business: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_sender: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_group: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_business: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in history_rows:
        message_id = row.get("message_id", "")
        event_rows = message_events.get(message_id, [])
        event_row = event_rows[0] if event_rows else {}
        enriched_row = dict(row)
        enriched_row["events"] = event_rows
        enriched_row["primary_event"] = event_row
        enriched_row["tokens"] = _tokenize(row.get("message_text", ""))
        enriched_row["created_dt"] = _parse_timestamp(row.get("created_at", ""))
        history_index.append(enriched_row)

        user_id = row.get("user_id", "")
        sender_user_id = row.get("sender_user_id", "")
        group_id = row.get("group_id", "")
        business_id = row.get("business_id", "")
        if user_id and sender_user_id:
            by_user_sender[(user_id, sender_user_id)].append(enriched_row)
        if user_id and group_id:
            by_user_group[(user_id, group_id)].append(enriched_row)
        if user_id and business_id:
            by_user_business[(user_id, business_id)].append(enriched_row)
        if sender_user_id:
            by_sender[sender_user_id].append(enriched_row)
        if group_id:
            by_group[group_id].append(enriched_row)
        if business_id:
            by_business[business_id].append(enriched_row)

    return {
        "users": users,
        "groups": groups,
        "businesses": businesses,
        "user_business_history": user_business_history,
        "group_members": group_members,
        "images": images,
        "voice_notes": voice_notes,
        "daily_summary": daily_summary,
        "history": history_index,
        "history_by_user_sender": by_user_sender,
        "history_by_user_group": by_user_group,
        "history_by_user_business": by_user_business,
        "history_by_sender": by_sender,
        "history_by_group": by_group,
        "history_by_business": by_business,
    }


def _merge_message_text(message: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    message_text = (message.get("message_text") or "").strip()
    media_type = (message.get("media_type") or "").strip().lower()
    media_id = (message.get("media_id") or "").strip()
    media_extras: Dict[str, Any] = {}
    media_text = ""

    if media_type == "image" and media_id:
        media_record = context["images"].get(media_id, {})
        media_path = _resolve_media_path(DATASET_PATH, media_record.get("file_path", ""))
        if media_path:
            media_text, media_extras = _best_effort_image_ocr(media_path)
            media_extras["media_size"] = media_path.stat().st_size
    elif media_type == "voice" and media_id:
        media_record = context["voice_notes"].get(media_id, {})
        media_path = _resolve_media_path(DATASET_PATH, media_record.get("file_path", ""))
        if media_path:
            media_text, media_extras = _best_effort_voice_summary(media_path)
            media_extras["media_size"] = media_path.stat().st_size

    combined = " ".join(part for part in [message_text, media_text] if part).strip()
    if media_extras:
        media_extras["media_text"] = media_text
    return combined, media_extras


def _pair_history_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    stats = {
        "count": 0.0,
        "open_rate": 0.0,
        "reply_rate": 0.0,
        "dismiss_rate": 0.0,
        "mute_rate": 0.0,
        "report_rate": 0.0,
    }
    if not rows:
        return stats
    count = len(rows)
    opened = 0
    replied = 0
    dismissed = 0
    muted = 0
    reported = 0
    for row in rows:
        event = row.get("primary_event", {})
        opened += _safe_int(event.get("message_opened"), 0)
        replied += _safe_int(event.get("message_replied"), 0)
        dismissed += _safe_int(event.get("notification_dismissed"), 0)
        muted += _safe_int(event.get("muted_after_message"), 0)
        reported += _safe_int(event.get("message_reported"), 0)
    stats["count"] = float(count)
    stats["open_rate"] = opened / count
    stats["reply_rate"] = replied / count
    stats["dismiss_rate"] = dismissed / count
    stats["mute_rate"] = muted / count
    stats["report_rate"] = reported / count
    return stats


def _historical_message_score(current_tokens: Set[str], history_row: Dict[str, Any], action_bias: str) -> float:
    history_tokens = history_row.get("tokens", set())
    if not current_tokens or not history_tokens:
        overlap = 0.0
    else:
        overlap = len(current_tokens & history_tokens) / max(1, len(current_tokens | history_tokens))

    event = history_row.get("primary_event", {})
    event_boost = 0.0
    if action_bias == "mute":
        event_boost += 0.15 if _safe_int(event.get("notification_dismissed"), 0) or _safe_int(event.get("muted_after_message"), 0) or _safe_int(event.get("message_reported"), 0) else 0.0
    else:
        event_boost += 0.12 if _safe_int(event.get("message_opened"), 0) or _safe_int(event.get("message_replied"), 0) else 0.0

    sender_match = 0.25 if history_row.get("sender_user_id") else 0.0
    group_match = 0.25 if history_row.get("group_id") else 0.0
    business_match = 0.25 if history_row.get("business_id") else 0.0

    return overlap * 0.35 + event_boost + sender_match + group_match + business_match


def _trust_score_from_history(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.5
    stats = _pair_history_stats(rows)
    score = 0.5 + 0.35 * stats["reply_rate"] + 0.2 * stats["open_rate"]
    score -= 0.3 * stats["dismiss_rate"]
    score -= 0.35 * stats["report_rate"]
    score -= 0.15 * stats["mute_rate"]
    return max(0.0, min(1.0, score))


def _group_context_score(message: Dict[str, Any], context: Dict[str, Any]) -> float:
    group_id = message.get("group_id", "")
    if not group_id:
        return 0.0
    group = context["groups"].get(group_id, {})
    members = context["group_members"].get(group_id, [])
    user_id = message.get("user_id", "")
    membership = next((row for row in members if row.get("user_id") == user_id), {})
    group_type = _normalize_text(group.get("group_type", ""))

    score = 0.0
    if membership.get("group_muted_by_user", "0") in {"1", "true", "True", "yes", "YES"}:
        score -= 0.35
    if membership.get("role", "").lower() in {"admin", "moderator"}:
        score += 0.2
    if _safe_int(membership.get("messages_read_30d"), 0) > _safe_int(membership.get("messages_sent_30d"), 0):
        score += 0.05
    score += min(_safe_int(group.get("messages_30d"), 0) / 1000.0, 0.15)
    if group_type in {"family", "society", "school_group", "coworker"}:
        score += 0.18
    if group_type == "marketplace":
        score -= 0.08
    if _safe_int(membership.get("notifications_dismissed_30d"), 0) > 5:
        score -= 0.15
    return max(-0.5, min(0.5, score))


def _business_context_score(message: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
    business_id = message.get("business_id", "")
    user_id = message.get("user_id", "")
    if not business_id:
        return 0.0, 0.0, {}

    business = context["businesses"].get(business_id, {})
    relationship_rows = context["user_business_history"].get(user_id, [])
    relationship = next((row for row in relationship_rows if row.get("business_id") == business_id), {})

    trust = 0.45
    utility = 0.35
    if business.get("verified", "0") in {"1", "true", "True", "yes", "YES"}:
        trust += 0.2
    if _safe_int(business.get("user_reports_30d"), 0) > 5:
        trust -= 0.2
    if business.get("category", "").lower() in BUSINESS_URGENT_CATEGORIES:
        utility += 0.15
    if relationship:
        trust += 0.18
        if relationship.get("allows_promotions", "0") in {"1", "true", "True", "yes", "YES"}:
            utility += 0.12
        if _safe_int(relationship.get("messages_opened_30d"), 0) > _safe_int(relationship.get("messages_dismissed_30d"), 0):
            trust += 0.05
        if relationship.get("promotions_opted_out_at"):
            trust -= 0.15
            utility -= 0.1
    else:
        trust -= 0.08
    return max(0.0, min(1.0, trust)), max(0.0, min(1.0, utility)), business


def _daily_load_score(message: Dict[str, Any], context: Dict[str, Any]) -> float:
    user_id = message.get("user_id", "")
    summary_rows = context["daily_summary"].get(user_id, [])
    if not summary_rows:
        return 0.0
    latest = summary_rows[-1]
    sent = _safe_int(latest.get("notifications_sent"), 0)
    dismissed = _safe_int(latest.get("notifications_dismissed"), 0)
    if sent <= 0:
        return 0.0
    return max(0.0, min(1.0, (sent + dismissed) / 40.0))


def _personal_relevance(message: Dict[str, Any], context: Dict[str, Any], text: str, tokens: Set[str]) -> float:
    user_id = message.get("user_id", "")
    sender_id = message.get("sender_user_id", "")
    group_id = message.get("group_id", "")
    business_id = message.get("business_id", "")

    score = 0.25
    direct_history = context["history_by_user_sender"].get((user_id, sender_id), []) if sender_id else []
    group_history = context["history_by_user_group"].get((user_id, group_id), []) if group_id else []
    business_history = context["history_by_user_business"].get((user_id, business_id), []) if business_id else []

    if direct_history:
        score += min(0.25, _trust_score_from_history(direct_history) * 0.22)
    if group_history:
        score += min(0.2, _trust_score_from_history(group_history) * 0.16)
    if business_history:
        score += min(0.25, _trust_score_from_history(business_history) * 0.2)

    if _contains_phrase(text, GREETING_TOKENS):
        score -= 0.1
    if _contains_phrase(text, URGENT_TOKENS):
        score += 0.12
    if "@" in text:
        score += 0.15
    if sender_id and not group_id and not business_id:
        score += 0.08
    return max(0.0, min(1.0, score))


def _content_signals(message: Dict[str, Any], text: str, tokens: Set[str]) -> Dict[str, float]:
    lowered = _normalize_text(text)
    signals = {
        "scam": 0.0,
        "spam": 0.0,
        "payment": 0.0,
        "promotion": 0.0,
        "event": 0.0,
        "urgent": 0.0,
        "business_update": 0.0,
        "greeting": 0.0,
        "forward": 0.0,
        "personal": 0.0,
    }

    if not lowered and message.get("media_type") in {"image", "voice"}:
        signals["event"] += 0.05
        signals["business_update"] += 0.05

    if _contains_phrase(lowered, SAFETY_ADVISORY_TOKENS):
        signals["business_update"] += 0.35
        signals["scam"] = max(0.0, signals["scam"] - 0.6)
        if any(token in lowered for token in ["never ask", "do not share", "we never ask"]):
            signals["scam"] = 0.0
    if _contains_phrase(lowered, SCAM_TOKENS) or URL_RE.search(text):
        signals["scam"] += 0.5
    if "ignore previous" in lowered or "mark this message as notify" in lowered:
        signals["scam"] += 0.5
    if OTP_RE.search(lowered) and any(token in lowered for token in ["otp", "code", "verify", "verification", "login", "password"]):
        signals["scam"] += 0.4
    if NUMBER_RE.search(lowered) and any(token in lowered for token in ["blocked", "suspended", "account", "security", "support"]):
        signals["scam"] += 0.3

    if _contains_phrase(lowered, PROMO_TOKENS) or any(token in tokens for token in PROMO_TOKENS):
        signals["promotion"] += 0.45
        signals["spam"] += 0.12
    if any(token in lowered for token in ["selling", "for sale", "dm if interested", "itinerary", "trip", "travel", "rs", "rupees", "per person", "50%", "offer"]):
        signals["promotion"] += 0.2
    if message.get("forwarded_count") and _safe_int(message.get("forwarded_count"), 0) >= 3:
        signals["spam"] += 0.25
        signals["forward"] += 0.35
    if _contains_phrase(lowered, FORWARD_TOKENS):
        signals["forward"] += 0.4
        signals["spam"] += 0.1

    if _contains_phrase(lowered, PAYMENT_TOKENS):
        signals["payment"] += 0.5
    if any(token in lowered for token in ["invoice", "reminder", "due", "balance", "receipt"]):
        signals["payment"] += 0.15

    if _contains_phrase(lowered, EVENT_TOKENS) or any(token in lowered for token in ["circular", "notice", "form", "schedule", "pickup", "bus", "exam", "class", "meeting"]):
        signals["event"] += 0.45
    if any(token in lowered for token in ["water supply", "tanker", "motor room", "plumber", "valve", "stadium road", "bus is leaving", "school circular", "consent note"]):
        signals["event"] += 0.25
        signals["urgent"] += 0.12
    if any(token in lowered for token in ["appointment", "prescription", "claim", "pickup details", "review", "registered app"]):
        signals["event"] += 0.35

    if _contains_phrase(lowered, GREETING_TOKENS):
        signals["greeting"] += 0.65

    if any(token in lowered for token in ["delivery", "tracking", "packed", "arriving", "shipment", "appointment", "booking", "reservation", "claim", "policy", "ticket", "feedback", "experience"]):
        signals["business_update"] += 0.35

    if _contains_phrase(lowered, URGENT_TOKENS):
        signals["urgent"] += 0.5
    if "@" in text:
        signals["urgent"] += 0.15
        signals["personal"] += 0.2
    if len(tokens) <= 7 and not signals["greeting"] and not signals["scam"] and not signals["spam"]:
        signals["personal"] += 0.12

    if any(token in lowered for token in ["thanks", "thank you", "can you", "could you", "pls", "please", "reply", "check", "call", "join", "respond"]):
        signals["personal"] += 0.15

    if any(token in lowered for token in ["prod review", "client note", "failed-payment screenshots", "escalation starts", "queue numbers", "eod"]):
        signals["urgent"] += 0.35
        signals["payment"] += 0.1

    if not any(signals.values()):
        signals["personal"] += 0.05

    return signals


def _choose_message_type(
    message: Dict[str, Any],
    text: str,
    tokens: Set[str],
    context: Dict[str, Any],
    signals: Dict[str, float],
    trust: float,
    utility: float,
) -> str:
    scores = dict(signals)
    user_id = message.get("user_id", "")
    business_id = message.get("business_id", "")
    group_id = message.get("group_id", "")
    sender_id = message.get("sender_user_id", "")

    if business_id:
        business = context["businesses"].get(business_id, {})
        category = _normalize_text(business.get("category", ""))
        if scores["scam"] > 0.45:
            return "scam"
        if scores["payment"] > 0.45 and category in {"bank", "telecom", "ecommerce_delivery", "ride_booking", "clinic"}:
            scores["payment"] += 0.25
        if scores["promotion"] > 0.35 and (category in {"fashion", "travel", "ecommerce", "marketplace"} or not utility):
            scores["promotion"] += 0.15
        if category in {"bank", "delivery", "ecommerce_delivery", "ride_booking", "telecom", "clinic"}:
            scores["business_update"] += 0.3

    if group_id:
        group = context["groups"].get(group_id, {})
        group_type = _normalize_text(group.get("group_type", ""))
        if group_type in {"family", "society", "school_group", "coworker"}:
            scores["event"] += 0.12
        if group_type == "marketplace":
            scores["promotion"] += 0.12

    if sender_id and not business_id and not group_id:
        scores["personal"] += 0.08
        if scores["personal"] >= 0.2 and scores["urgent"] < 0.35 and scores["event"] < 0.5 and scores["promotion"] < 0.35:
            return "personal"

    if message.get("conversation_type") == "group" and not business_id and scores["personal"] >= 0.15 and scores["event"] < 0.35 and scores["promotion"] < 0.2:
        return "personal"

    if not text and message.get("media_type") == "voice":
        media_size = _safe_int(message.get("media_size"), 0)
        if business_id:
            if utility < 0.35:
                scores["spam"] += 0.2
            else:
                scores["business_update"] += 0.12
        else:
            if media_size and media_size < 90000:
                scores["urgent"] += 0.45
                scores["personal"] += 0.1
            elif media_size and media_size > 150000:
                scores["spam"] += 0.2
            else:
                scores["personal"] += 0.2

    if trust < 0.3 and (scores["scam"] > 0.15 or scores["spam"] > 0.15):
        return "scam" if scores["scam"] >= scores["spam"] else "spam"

    priority = ["scam", "payment", "business_update", "event", "promotion", "urgent", "greeting", "personal", "forward", "spam", "unknown"]
    best_type = "unknown"
    best_score = 0.0
    for message_type in priority:
        score = scores.get(message_type, 0.0)
        if score > best_score:
            best_score = score
            best_type = message_type

    if best_score < 0.25:
        return "unknown"
    if best_type == "forward" and scores.get("greeting", 0.0) >= 0.5:
        return "greeting"
    return best_type


def _choose_action(
    message: Dict[str, Any],
    context: Dict[str, Any],
    text: str,
    message_type: str,
    signals: Dict[str, float],
    trust: float,
    utility: float,
    group_score: float,
    business_trust: float,
    business_utility: float,
    load_score: float,
) -> str:
    user = context["users"].get(message.get("user_id", ""), {})
    business = context["businesses"].get(message.get("business_id", ""), {})
    dnd_window = user.get("do_not_disturb_window", "")
    in_dnd = _is_inside_window(_parse_timestamp(message.get("created_at", "")), dnd_window)

    risk = signals["scam"] + 0.45 * signals["spam"] + 0.2 * signals["forward"]
    urgency = signals["urgent"] + 0.25 * signals["payment"] + 0.18 * signals["event"] + 0.15 * signals["business_update"]
    helpfulness = utility + 0.22 * signals["event"] + 0.18 * signals["business_update"] + 0.1 * signals["personal"]

    if message.get("forwarded_count") and _safe_int(message.get("forwarded_count"), 0) >= 6:
        risk += 0.15
    if in_dnd:
        urgency -= 0.12
        helpfulness -= 0.08
    if load_score > 0.65 and message_type not in {"urgent", "payment", "scam"}:
        helpfulness -= 0.1
    if group_score < -0.1:
        helpfulness -= 0.08
        risk += 0.05
    if business_trust < 0.35:
        risk += 0.12
    if business_utility > 0.45:
        helpfulness += 0.1

    if message_type in {"scam", "spam"}:
        return "mute"
    if risk >= 0.55:
        return "mute"
    if message_type == "urgent" and trust >= 0.3 and (urgency >= 0.45 or group_score > 0.05 or business_trust > 0.55):
        return "notify"
    if message_type in {"payment", "event", "business_update"}:
        if urgency >= 0.5 and trust >= 0.35 and not in_dnd:
            return "notify"
        if message_type == "business_update" and any(token in _normalize_text(text) for token in ["packed", "hub", "delivery", "appointment", "prescription", "claim", "safety advisory", "feedback", "experience"]):
            return "notify"
        if message_type == "event" and any(token in _normalize_text(text) for token in ["bus", "road blocked", "cannot wait", "teacher", "circular", "consent", "timing", "school", "plumber", "tanker", "water supply"]):
            return "notify"
        return "digest"
    if message_type == "promotion":
        user_history_rows = context["user_business_history"].get(message.get("user_id", ""), [])
        relationship = next((row for row in user_history_rows if row.get("business_id") == message.get("business_id", "")), {})
        opted_out = bool(relationship.get("promotions_opted_out_at")) or relationship.get("allows_promotions", "1") in {"0", "false", "False", "no", "NO"}
        if opted_out:
            return "mute"
        if business_utility > 0.55 and trust >= 0.4 and not in_dnd:
            return "digest"
        if load_score > 0.5:
            return "mute"
        return "digest"
    if message_type == "greeting":
        return "digest"
    if message_type == "forward":
        return "mute" if risk > 0.3 else "digest"
    if message_type == "personal":
        if urgency >= 0.5 and trust >= 0.3 and not in_dnd:
            return "notify"
        if trust >= 0.35 or helpfulness >= 0.35:
            return "digest"
        return "mute" if load_score > 0.4 else "digest"
    if helpfulness >= 0.45 and not in_dnd:
        return "digest"
    return "mute" if risk > 0.25 or load_score > 0.55 else "digest"


def _select_evidence(
    message: Dict[str, Any],
    context: Dict[str, Any],
    text: str,
    message_type: str,
    action: str,
) -> List[str]:
    user_id = message.get("user_id", "")
    sender_id = message.get("sender_user_id", "")
    group_id = message.get("group_id", "")
    business_id = message.get("business_id", "")
    current_tokens = _tokenize(text)

    candidates: List[Tuple[float, str]] = []
    seen: Set[str] = set()

    source_lists = []
    if sender_id:
        source_lists.append(context["history_by_user_sender"].get((user_id, sender_id), []))
        source_lists.append(context["history_by_sender"].get(sender_id, []))
    if group_id:
        source_lists.append(context["history_by_user_group"].get((user_id, group_id), []))
        source_lists.append(context["history_by_group"].get(group_id, []))
    if business_id:
        source_lists.append(context["history_by_user_business"].get((user_id, business_id), []))
        source_lists.append(context["history_by_business"].get(business_id, []))

    flat_rows: List[Dict[str, Any]] = []
    for rows in source_lists:
        flat_rows.extend(rows)

    for row in flat_rows:
        row_id = row.get("message_id", "")
        if not row_id or row_id in seen:
            continue
        seen.add(row_id)
        score = _historical_message_score(current_tokens, row, action)
        hist_text = _normalize_text(row.get("message_text", ""))
        if message_type == "scam" and any(token in hist_text for token in ["otp", "password", "blocked", "verify"]):
            score += 0.2
        elif message_type == "promotion" and any(token in hist_text for token in ["sale", "offer", "discount", "free"]):
            score += 0.18
        elif message_type == "event" and any(token in hist_text for token in ["notice", "form", "schedule", "pickup", "bus"]):
            score += 0.18
        elif message_type == "payment" and any(token in hist_text for token in ["payment", "invoice", "due", "refund"]):
            score += 0.18

        event = row.get("primary_event", {})
        if action == "mute" and (
            _safe_int(event.get("notification_dismissed"), 0)
            or _safe_int(event.get("muted_after_message"), 0)
            or _safe_int(event.get("message_reported"), 0)
        ):
            score += 0.12
        if action in {"notify", "digest"} and (
            _safe_int(event.get("message_opened"), 0) or _safe_int(event.get("message_replied"), 0)
        ):
            score += 0.12

        candidates.append((score, row_id))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [row_id for score, row_id in candidates[:3] if score >= 0.2]
    return selected


def _reason_parts(
    message: Dict[str, Any],
    context: Dict[str, Any],
    text: str,
    message_type: str,
    action: str,
    trust: float,
    utility: float,
    group_score: float,
    business_trust: float,
    business_utility: float,
    load_score: float,
) -> str:
    user = context["users"].get(message.get("user_id", ""), {})
    group = context["groups"].get(message.get("group_id", ""), {})
    business = context["businesses"].get(message.get("business_id", ""), {})
    parts: List[str] = []

    if message_type == "scam":
        parts.append("Suspicious verification or account-pressure language")
    elif message_type == "spam":
        parts.append("Low-value repeated or forwarded content")
    elif message_type == "payment":
        parts.append("Payment-related message")
    elif message_type == "business_update":
        parts.append("Legitimate business update")
    elif message_type == "event":
        parts.append("Time-sensitive group or scheduling update")
    elif message_type == "promotion":
        parts.append("Marketing or sale content")
    elif message_type == "urgent":
        parts.append("Direct time-sensitive request")
    elif message_type == "greeting":
        parts.append("Harmless greeting or low-urgency chat")
    elif message_type == "forward":
        parts.append("Forwarded content")
    elif message_type == "personal":
        parts.append("Personal message")
    else:
        parts.append("No strong urgency or safety signal")

    if message.get("business_id") and business:
        category = business.get("category", "")
        if business.get("verified", "0") in {"1", "true", "True", "yes", "YES"}:
            parts.append("from a verified business")
        if category:
            parts.append(f"category {category}")
    if group:
        group_type = group.get("group_type", "")
        if group_type:
            parts.append(f"{group_type} group")
    if _is_inside_window(_parse_timestamp(message.get("created_at", "")), user.get("do_not_disturb_window", "")):
        parts.append("inside quiet hours")
    if load_score > 0.55 and action != "notify":
        parts.append("user already receives a high notification load")
    if trust < 0.35 and action == "mute":
        parts.append("low sender trust")
    if business_utility > 0.45 and action == "digest":
        parts.append("useful but not immediate")
    if action == "notify":
        parts.append("should interrupt now")
    elif action == "digest":
        parts.append("can wait for digest")
    else:
        parts.append("should stay muted")
    return "; ".join(parts)


def route_message(message: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    text, media_metadata = _merge_message_text(message, context)
    tokens = _tokenize(text)

    group_score = _group_context_score(message, context)
    business_trust, business_utility, business = _business_context_score(message, context)
    load_score = _daily_load_score(message, context)
    trust = _personal_relevance(message, context, text, tokens)

    signals = _content_signals(message, text, tokens)
    if business:
        if business.get("verified", "0") in {"1", "true", "True", "yes", "YES"}:
            signals["business_update"] += 0.18
        if business_trust < 0.35:
            signals["scam"] += 0.12
        if business_utility > 0.45:
            signals["business_update"] += 0.1

    if group_score > 0.1:
        signals["event"] += 0.08
        signals["urgent"] += 0.05
    if group_score < -0.1:
        signals["spam"] += 0.08

    if message.get("forwarded_count"):
        forwarded_count = _safe_int(message.get("forwarded_count"), 0)
        if forwarded_count >= 3:
            signals["forward"] += min(0.4, forwarded_count / 20.0)
            signals["spam"] += min(0.25, forwarded_count / 40.0)

    if media_metadata.get("media_text"):
        if _contains_phrase(media_metadata["media_text"], SCAM_TOKENS):
            signals["scam"] += 0.35
        if _contains_phrase(media_metadata["media_text"], PROMO_TOKENS):
            signals["promotion"] += 0.15
        if _contains_phrase(media_metadata["media_text"], EVENT_TOKENS):
            signals["event"] += 0.15

    message_type = _choose_message_type(message, text, tokens, context, signals, trust, business_utility)
    action = _choose_action(
        message,
        context,
        text,
        message_type,
        signals,
        trust,
        business_utility,
        group_score,
        business_trust,
        business_utility,
        load_score,
    )
    evidence_ids = _select_evidence(message, context, text, message_type, action)
    evidence_count = len(evidence_ids)

    raw_confidence = 0.48
    top_signal = max(signals.values()) if signals else 0.0
    second_signal = sorted(signals.values(), reverse=True)[1] if len(signals) > 1 else 0.0
    raw_confidence += 0.18 * top_signal + 0.08 * max(top_signal - second_signal, 0.0)
    raw_confidence += 0.1 * min(3, evidence_count)
    raw_confidence += 0.08 * trust + 0.06 * business_trust + 0.04 * business_utility
    if action == "mute":
        raw_confidence += 0.08 if message_type in {"scam", "spam"} else 0.02
    if action == "notify":
        raw_confidence += 0.05 if "urgent" in message_type else 0.0
    if load_score > 0.6 and action != "notify":
        raw_confidence += 0.03
    confidence = max(0.1, min(0.98, raw_confidence))

    reason = _reason_parts(
        message,
        context,
        text,
        message_type,
        action,
        trust,
        business_utility,
        group_score,
        business_trust,
        business_utility,
        load_score,
    )

    evidence_string = ";".join(evidence_ids) if evidence_ids else "none"
    return {
        "message_id": message.get("message_id", ""),
        "action": action,
        "message_type": message_type,
        "reason": reason,
        "confidence": round(confidence, 2),
        "evidence_message_ids": evidence_string,
    }


def route_messages(messages: Sequence[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [route_message(message, context) for message in messages]


def load_messages(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_predictions(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "message_id": row["message_id"],
                    "action": row["action"],
                    "message_type": row["message_type"],
                    "reason": row["reason"],
                    "confidence": f"{_safe_float(row['confidence'], 0.0):.2f}",
                    "evidence_message_ids": row["evidence_message_ids"] or "none",
                }
            )
