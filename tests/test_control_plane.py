from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from control_plane.config import AppConfig
from control_plane.data import assess_quality, infer_mapping, normalize_dataframe
from control_plane.policy import evaluate_policy
from control_plane.normalizer import detect_language, normalize_ticket
from control_plane.service import ControlPlane


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticket ID": "A-1",
                "Title": "Unauthorized account access",
                "Description": "We suspect a security breach and are locked out.",
                "Lang": "en",
                "Queue": "Security",
                "Priority": "High",
                "Type": "Incident",
                "Tags": "security, account access",
            },
            {
                "Ticket ID": "A-2",
                "Title": "How to upgrade",
                "Description": "How can I upgrade our plan when convenient?",
                "Lang": "en",
                "Queue": "Customer Success",
                "Priority": "Low",
                "Type": "Question",
                "Tags": "onboarding",
            },
        ]
    )


def test_mapping_normalization_and_quality() -> None:
    raw = sample_frame()
    mapping = infer_mapping(list(raw.columns))
    assert mapping["Ticket ID"] == "ticket_id"
    assert mapping["Description"] == "body"
    normalized = normalize_dataframe(raw, mapping)
    assert list(normalized.ticket_id) == ["A-1", "A-2"]
    assert normalized.loc[0, "priority_actual"] == "High"
    quality = assess_quality(raw, mapping)
    assert quality.is_valid
    assert quality.valid_rows == 2


def test_kaggle_style_split_tags_and_priority_are_normalized() -> None:
    raw = pd.DataFrame([{"id": "K-1", "subject": "Access", "body": "Cannot login", "priority": "3 (Critical)", "tag_1": "Security", "Tag 2": "Account Access"}])
    normalized = normalize_dataframe(raw)
    assert normalized.loc[0, "priority_actual"] == "High"
    assert normalized.loc[0, "tags"] == "Security, Account Access"


def test_local_language_normalization_preserves_original() -> None:
    ticket = {"subject": "Konto gesperrt", "body": "Die Anmeldung ist fehlgeschlagen. Bitte helfen Sie.", "language": "de", "queue_actual": "IT Support", "priority_actual": "High", "type_actual": "Incident", "tags": "Account"}
    result = normalize_ticket(ticket, ["en", "de"])
    assert ticket["subject"] == "Konto gesperrt"
    assert result.subject_normalized == "account locked"
    assert result.normalization_status == "normalized_with_fallback"
    assert result.fallback_used
    assert detect_language(ticket["body"])[0] == "de"


def test_policy_forces_review_for_risk_tag() -> None:
    config = AppConfig(review_threshold=0.5, warning_threshold=0.6)
    ticket = {"subject": "Card issue", "body": "Please investigate", "tags": "security", "language": "en", "queue_actual": "Security"}
    prediction = {
        "queue_pred": "Security", "priority_pred": "Medium", "type_pred": "Request",
        "confidence_queue": 0.95, "confidence_priority": 0.95, "confidence_type": 0.95,
    }
    result = evaluate_policy(ticket, prediction, config)
    assert result.final_action == "human_review"
    assert any(rule["rule_id"] == "TAG_RISK" for rule in result.triggered_rules)


def test_end_to_end_review_audit_and_export(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path / "control.db")
    count, quality = plane.ingest(sample_frame())
    assert count == 2 and quality.is_valid
    assert plane.process() == 2
    tickets = plane.ticket_view()
    assert tickets.queue_pred.notna().all()
    assert tickets.final_action.notna().all()
    queue = plane.review_queue()
    assert "A-1" in queue.ticket_id.tolist()
    row = queue[queue.ticket_id == "A-1"].iloc[0]
    plane.review("A-1", "reroute", "Casey", "Escalate to incident response", "Incident Response", "High", row.type_pred)
    trace = plane.ticket_trace("A-1")
    assert trace["reviews"][-1]["reviewer_decision"] == "reroute"
    assert trace["decision"]["destination_queue"] == "Incident Response"
    assert {event["stage"] for event in trace["audit"]} >= {"ingestion", "prediction", "policy", "review"}
    exported = plane.export_actions()
    assert "A-1" in exported.ticket_id.tolist()
    assert exported.loc[exported.ticket_id == "A-1", "final_queue"].iloc[0] == "Incident Response"
    payload = plane.jira_payload("A-1")
    assert payload["fields"]["routingQueue"] == "Incident Response"


def test_configuration_persists_and_changes_routes(tmp_path: Path) -> None:
    path = tmp_path / "control.db"
    plane = ControlPlane(path)
    plane.ingest(sample_frame())
    plane.process()
    config = plane.config
    config.review_threshold = 0.9
    config.default_reviewer = "Morgan"
    plane.save_config(config)
    reopened = ControlPlane(path)
    assert reopened.config.review_threshold == 0.9
    assert reopened.config.default_reviewer == "Morgan"
    assert "human_review" in set(reopened.ticket_view().final_action)


def test_audit_payloads_are_valid_json(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path / "control.db")
    plane.ingest(sample_frame())
    plane.process()
    for event in plane.ticket_trace("A-1")["audit"]:
        assert isinstance(json.loads(event["payload"]), dict)
