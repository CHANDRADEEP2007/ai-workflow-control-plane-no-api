from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig


@dataclass(slots=True)
class PolicyResult:
    final_action: str
    destination_queue: str
    triggered_rules: list[dict[str, Any]]
    rationale: str


def evaluate_policy(ticket: dict[str, Any], prediction: dict[str, Any], config: AppConfig) -> PolicyResult:
    rules: list[dict[str, Any]] = []
    confidences = [float(prediction[f"confidence_{field}"]) for field in ("queue", "priority", "type")]
    minimum = min(confidences)
    tags = {tag.strip().lower() for tag in str(ticket.get("tags", "")).replace(";", ",").split(",") if tag.strip()}
    text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()
    normalization_status = str(ticket.get("normalization_status", "success"))
    language_source = str(ticket.get("language_source", ticket.get("language", "en"))).lower()
    if normalization_status in {"failed", "unsupported"}:
        rules.append({"rule_id": "NORMALIZATION_FAILED", "name": "Language normalization failed", "severity": "review", "detail": f"Status {normalization_status}; source language {language_source}"})
    elif normalization_status == "normalized_with_fallback":
        severity = "review" if prediction["priority_pred"] == "High" else "warning"
        rules.append({"rule_id": "NORMALIZATION_FALLBACK", "name": "Local normalization fallback", "severity": severity, "detail": f"Offline lexicon fallback used for {language_source}"})
    if minimum < config.review_threshold:
        rules.append({"rule_id": "CONF_LOW", "name": "Low confidence", "severity": "review", "detail": f"Minimum confidence {minimum:.0%} < {config.review_threshold:.0%}"})
    elif minimum < config.warning_threshold:
        rules.append({"rule_id": "CONF_WARN", "name": "Moderate confidence", "severity": "warning", "detail": f"Minimum confidence {minimum:.0%} < {config.warning_threshold:.0%}"})
    restricted = sorted(tags.intersection(tag.lower() for tag in config.restricted_tags))
    if restricted:
        rules.append({"rule_id": "TAG_RISK", "name": "Restricted tag", "severity": "review", "detail": ", ".join(restricted)})
    keywords = [keyword for keyword in config.risk_keywords if keyword.lower() in text]
    if keywords:
        rules.append({"rule_id": "KEYWORD_RISK", "name": "High-risk keyword", "severity": "review", "detail": ", ".join(keywords)})
    if config.high_priority_requires_review and prediction["priority_pred"] == "High" and minimum < 0.9:
        rules.append({"rule_id": "HIGH_PRIORITY", "name": "High priority safeguard", "severity": "review", "detail": "High priority prediction requires stronger confidence"})
    language = language_source
    destination = prediction["queue_pred"]
    if language not in {lang.lower() for lang in config.supported_languages}:
        destination = config.language_queue
        rules.append({"rule_id": "LANGUAGE", "name": "Unsupported language", "severity": "warning", "detail": f"{language} routes to {destination}"})
    actual = str(ticket.get("queue_actual", ""))
    if config.evaluation_mode and config.baseline_change_review and actual and actual != prediction["queue_pred"]:
        rules.append({"rule_id": "BASELINE_CHANGE", "name": "Baseline disagreement", "severity": "review", "detail": f"Actual {actual}; predicted {prediction['queue_pred']}"})
    severities = {rule["severity"] for rule in rules}
    action = "human_review" if "review" in severities else "auto_route_warning" if "warning" in severities else "auto_route"
    rationale = "No policy exceptions; safe to auto-route." if not rules else " | ".join(rule["detail"] for rule in rules)
    return PolicyResult(action, destination, rules, rationale)
