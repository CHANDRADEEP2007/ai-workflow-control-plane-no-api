from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class NormalizationResult:
    subject_normalized: str
    body_normalized: str
    language_source: str
    normalization_method: str
    normalization_status: str
    normalization_confidence: float
    fallback_used: bool
    translation_model_version: str
    normalized_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GERMAN_MARKERS = {
    "der", "die", "das", "und", "nicht", "ich", "wir", "bitte", "problem",
    "fehler", "konto", "anmeldung", "unterstützung", "dringend", "sicherheit",
}

GERMAN_PHRASES = {
    "sehr geehrtes support-team": "dear support team",
    "vielen dank": "thank you",
    "ich möchte": "i would like",
    "ich kann mich nicht anmelden": "i cannot sign in",
    "nicht funktioniert": "not working",
    "funktioniert nicht": "not working",
    "nicht verfügbar": "unavailable",
    "bitte helfen sie": "please help",
    "so schnell wie möglich": "as soon as possible",
    "dringend": "urgent",
    "kritisch": "critical",
    "sicherheitsvorfall": "security incident",
    "datenverletzung": "data breach",
    "cyberattacke": "cyberattack",
    "ungewöhnliche aktivitäten": "unusual activity",
    "konto gesperrt": "account locked",
    "gesperrt": "locked",
    "anmeldung": "login",
    "kennwort": "password",
    "passwort": "password",
    "konto": "account",
    "rechnung": "invoice",
    "zahlung": "payment",
    "rückerstattung": "refund",
    "abbuchung": "charge",
    "abonnement": "subscription",
    "kündigen": "cancel",
    "fehler": "error",
    "problem": "problem",
    "ausfall": "outage",
    "störung": "disruption",
    "langsam": "slow",
    "fehlgeschlagen": "failed",
    "zugriff": "access",
    "berechtigung": "permission",
    "benutzer": "user",
    "mitarbeiter": "employee",
    "software": "software",
    "gerät": "device",
    "geräte": "devices",
    "netzwerk": "network",
    "server": "server",
    "speicher": "storage",
    "cloud": "cloud",
    "aktualisierung": "update",
    "installation": "installation",
    "integration": "integration",
    "anfrage": "request",
    "änderung": "change",
    "hilfe": "help",
    "unterstützung": "support",
    "frage": "question",
    "produkt": "product",
    "bestellung": "order",
    "lieferung": "delivery",
    "rückgabe": "return",
    "umtausch": "exchange",
}


def detect_language(text: str) -> tuple[str, float]:
    """Small, dependency-free fallback used only when the dataset has no language value."""
    lowered = text.lower()
    words = set(re.findall(r"[a-zäöüß]+", lowered))
    score = len(words.intersection(GERMAN_MARKERS)) + sum(char in lowered for char in "äöüß")
    return ("de", min(0.95, 0.55 + score * 0.08)) if score >= 2 else ("en", 0.65)


def _lexicon_normalize(text: str) -> tuple[str, float]:
    normalized = text.lower()
    hits = 0
    for source, target in sorted(GERMAN_PHRASES.items(), key=lambda item: len(item[0]), reverse=True):
        count = normalized.count(source)
        if count:
            normalized = normalized.replace(source, target)
            hits += count
    tokens = max(1, len(re.findall(r"\w+", text)))
    confidence = min(0.88, 0.45 + min(0.43, hits * 5 / tokens))
    return normalized.strip(), round(confidence, 3)


def normalize_ticket(ticket: dict[str, Any], supported_languages: list[str]) -> NormalizationResult:
    subject = str(ticket.get("subject", ""))
    body = str(ticket.get("body", ""))
    language = str(ticket.get("language", "")).strip().lower()
    if not language:
        language, _ = detect_language(f"{subject} {body}")
    now = datetime.now(timezone.utc).isoformat()
    if language == "en":
        return NormalizationResult(subject, body, language, "identity", "success", 1.0, False, "identity-v1", now)
    if language not in {value.lower() for value in supported_languages}:
        return NormalizationResult(subject, body, language, "preserve_original", "unsupported", 0.0, True, "local-normalizer-v1", now)
    if language == "de":
        subject_en, subject_confidence = _lexicon_normalize(subject)
        body_en, body_confidence = _lexicon_normalize(body)
        context = (
            f"\n\nEnglish routing context from source metadata: tags {ticket.get('tags', '')}."
        )
        confidence = round((subject_confidence + body_confidence) / 2, 3)
        return NormalizationResult(
            subject_en, body_en + context, language, "local_lexicon_and_metadata",
            "normalized_with_fallback", confidence, True, "de-en-lexicon-v1", now,
        )
    return NormalizationResult(subject, body, language, "preserve_original", "failed", 0.0, True, "local-normalizer-v1", now)
