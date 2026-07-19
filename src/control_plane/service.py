from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .classifier import LocalTicketClassifier, is_evaluation_holdout
from .config import AppConfig
from .data import DataQuality, assess_quality, normalize_dataframe, read_csv
from .db import Database
from .policy import evaluate_policy
from .normalizer import normalize_ticket


class ControlPlane:
    def __init__(self, db_path: str | Path = "data/control_plane.db") -> None:
        self.db = Database(db_path)

    @property
    def config(self) -> AppConfig:
        return self.db.get_config()

    def ingest(self, source: Any, mapping: dict[str, str] | None = None) -> tuple[int, DataQuality]:
        frame = source if isinstance(source, pd.DataFrame) else read_csv(source)
        quality = assess_quality(frame, mapping)
        normalized = normalize_dataframe(frame, mapping)
        valid = normalized[(normalized["subject"] != "") & (normalized["body"] != "")]
        count = self.db.upsert_tickets(valid)
        return count, quality

    def process(self, ticket_ids: list[str] | None = None) -> int:
        config = self.config
        self.normalize_pending()
        training_rows = self.db.query(
            "SELECT t.*, n.subject_normalized, n.body_normalized, n.language_source, n.normalization_method, "
            "n.normalization_status, n.normalization_confidence, n.fallback_used, n.translation_model_version "
            "FROM tickets t JOIN normalizations n USING(ticket_id)"
        )
        if not training_rows:
            return 0
        classifier = LocalTicketClassifier(config.model_version)
        classifier.fit(training_rows)
        selected = set(ticket_ids or [])
        tickets = training_rows if not ticket_ids else [row for row in training_rows if row["ticket_id"] in selected]
        predictions = classifier.predict(tickets)
        batch: list[tuple[dict[str, Any], dict[str, Any], Any]] = []
        for ticket, prediction in zip(tickets, predictions):
            policy = evaluate_policy(ticket, prediction, config)
            batch.append((ticket, prediction, policy))
            if len(batch) >= 1000:
                self.db.save_processing_batch(batch, config)
                batch.clear()
        self.db.save_processing_batch(batch, config)
        return len(tickets)

    def normalize_pending(self, force: bool = False) -> int:
        where = "" if force else "WHERE n.ticket_id IS NULL"
        tickets = self.db.query(
            "SELECT t.* FROM tickets t LEFT JOIN normalizations n USING(ticket_id) " + where
        )
        config = self.config
        batch = [(ticket["ticket_id"], normalize_ticket(ticket, config.supported_languages).to_dict()) for ticket in tickets]
        self.db.save_normalizations(batch)
        return len(batch)

    def reprocess(self) -> int:
        return self.process()

    def overview(self) -> dict[str, Any]:
        tickets = self.db.count("tickets")
        decisions = self.db.dataframe("SELECT final_action, status FROM routing_decisions")
        return {
            "tickets": tickets,
            "processed": len(decisions),
            "pending_review": int((decisions.get("status", pd.Series(dtype=str)) == "pending_review").sum()),
            "auto_route_rate": 0.0 if decisions.empty else float(decisions["final_action"].str.startswith("auto_route").mean()),
            "audit_events": self.db.count("audit_log"),
        }

    def setup_status(self) -> dict[str, bool]:
        tickets = self.db.count("tickets")
        loaded = tickets > 0
        processed = loaded and self.db.count("predictions") == tickets and self.db.count("normalizations") == tickets
        confirmed = self.db.get_state("policy_confirmed") == "true"
        if processed and not self.db.get_state("setup_started") and not self.db.get_state("setup_complete") and not self.db.get_state("policy_confirmed"):
            # Existing installations were already configured before the two-mode redesign.
            confirmed = True
            self.db.set_state("policy_confirmed", "true")
            self.db.set_state("setup_complete", "true")
        complete = loaded and processed and confirmed
        return {"loaded": loaded, "processed": processed, "policy_confirmed": confirmed, "complete": complete}

    def confirm_policy_setup(self) -> None:
        self.db.set_state("setup_started", "true")
        self.db.set_state("policy_confirmed", "true")
        self.db.set_state("setup_complete", "true")
        self.db.audit(None, "setup", "operator", "setup_completed", self.setup_status())

    def start_new_dataset(self) -> None:
        self.db.reset(reset_setup=True)
        self.db.set_state("setup_started", "true")
        self.db.audit(None, "setup", "operator", "new_dataset_started", {})

    def ticket_view(self) -> pd.DataFrame:
        return self.db.dataframe(
            "SELECT t.*, t.subject subject_original, t.body body_original, n.subject_normalized, n.body_normalized, "
            "n.language_source, n.normalization_method, n.normalization_status, n.normalization_confidence, n.fallback_used, n.translation_model_version, "
            "p.queue_pred, p.priority_pred, p.type_pred, p.confidence_queue, p.confidence_priority, p.confidence_type, p.explanation_text, p.model_version, "
            "d.final_action, d.destination_queue, d.rationale, d.status "
            "FROM tickets t LEFT JOIN normalizations n USING(ticket_id) LEFT JOIN predictions p USING(ticket_id) "
            "LEFT JOIN routing_decisions d USING(ticket_id) ORDER BY t.created_at DESC"
        )

    def review_queue(self) -> pd.DataFrame:
        return self.db.dataframe(
            "SELECT t.ticket_id, t.subject, t.body, t.tags, t.language, t.created_at, n.subject_normalized, n.body_normalized, "
            "n.language_source, n.normalization_method, n.normalization_status, n.normalization_confidence, n.fallback_used, "
            "p.queue_pred, p.priority_pred, p.type_pred, p.confidence_queue, p.confidence_priority, "
            "p.confidence_type, p.explanation_text, p.model_version, p.predicted_at, "
            "d.final_action, d.destination_queue, d.rationale, d.status "
            "FROM tickets t JOIN normalizations n USING(ticket_id) JOIN predictions p USING(ticket_id) JOIN routing_decisions d USING(ticket_id) "
            "WHERE d.status='pending_review' ORDER BY CASE p.priority_pred WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END, t.created_at"
        )

    def ticket_trace(self, ticket_id: str) -> dict[str, Any]:
        ticket = self.db.query("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,))
        prediction = self.db.query("SELECT * FROM predictions WHERE ticket_id=?", (ticket_id,))
        normalization = self.db.query("SELECT * FROM normalizations WHERE ticket_id=?", (ticket_id,))
        decision = self.db.query("SELECT * FROM routing_decisions WHERE ticket_id=?", (ticket_id,))
        rules = self.db.query("SELECT * FROM policy_events WHERE ticket_id=? ORDER BY event_time", (ticket_id,))
        reviews = self.db.query("SELECT * FROM review_actions WHERE ticket_id=? ORDER BY reviewed_at", (ticket_id,))
        audit = self.db.query("SELECT * FROM audit_log WHERE ticket_id=? ORDER BY event_time, audit_id", (ticket_id,))
        return {"ticket": ticket[0] if ticket else {}, "normalization": normalization[0] if normalization else {}, "prediction": prediction[0] if prediction else {}, "decision": decision[0] if decision else {}, "rules": rules, "reviews": reviews, "audit": audit}

    def analytics(self) -> dict[str, Any]:
        data = self.ticket_view()
        reviews = self.db.dataframe("SELECT * FROM review_actions ORDER BY reviewed_at")
        rules = self.db.dataframe("SELECT rule_id, rule_name, event_time FROM policy_events")
        if data.empty:
            return {"data": data, "reviews": reviews, "rules": rules, "queue_accuracy": 0.0, "priority_accuracy": 0.0, "type_accuracy": 0.0, "review_rate": 0.0, "override_rate": 0.0, "audit_completeness": 0.0, "translation_coverage": 0.0, "translation_failure_rate": 0.0, "reviewer_handling_minutes": 0.0}
        processed = data[data["queue_pred"].notna()].copy()
        holdout = processed[processed["ticket_id"].map(is_evaluation_holdout)]
        evaluation = holdout if not holdout.empty else processed
        accuracy = lambda actual, pred: 0.0 if evaluation.empty else float((evaluation[actual].str.lower() == evaluation[pred].str.lower()).mean())
        review_rate = float((processed["final_action"] == "human_review").mean()) if not processed.empty else 0.0
        override_rate = float(reviews["reviewer_decision"].isin(["reject", "reroute"]).mean()) if not reviews.empty else 0.0
        complete_ids = self.db.dataframe("SELECT ticket_id, COUNT(DISTINCT stage) stages FROM audit_log WHERE ticket_id IS NOT NULL GROUP BY ticket_id")
        completeness = float((complete_ids["stages"] >= 4).mean()) if not complete_ids.empty else 0.0
        non_english = processed[processed["language_source"].fillna("en") != "en"]
        successful = {"success", "normalized_with_fallback"}
        coverage = float(non_english["normalization_status"].isin(successful).mean()) if not non_english.empty else 1.0
        failure_rate = float(non_english["normalization_status"].isin(["failed", "unsupported"]).mean()) if not non_english.empty else 0.0
        handling = self.db.dataframe(
            "SELECT (julianday(r.reviewed_at)-julianday(d.decided_at))*1440 minutes FROM review_actions r JOIN routing_decisions d USING(ticket_id)"
        )
        handling_minutes = float(handling["minutes"].clip(lower=0).mean()) if not handling.empty else 0.0
        return {"data": processed, "reviews": reviews, "rules": rules, "queue_accuracy": accuracy("queue_actual", "queue_pred"), "priority_accuracy": accuracy("priority_actual", "priority_pred"), "type_accuracy": accuracy("type_actual", "type_pred"), "review_rate": review_rate, "override_rate": override_rate, "audit_completeness": completeness, "translation_coverage": coverage, "translation_failure_rate": failure_rate, "reviewer_handling_minutes": handling_minutes}

    def review(self, ticket_id: str, decision: str, reviewer: str, notes: str, final_queue: str, final_priority: str, final_type: str) -> None:
        if decision not in {"approve", "reject", "reroute"}:
            raise ValueError("Unsupported reviewer decision")
        self.db.save_review(ticket_id, reviewer, decision, notes, final_queue, final_priority, final_type)

    def save_config(self, config: AppConfig, reprocess: bool = True) -> None:
        previous = self.config
        self.db.save_config(config)
        if reprocess and self.db.count("tickets"):
            if previous.supported_languages != config.supported_languages:
                self.normalize_pending(force=True)
            self.reprocess()

    def export_actions(self) -> pd.DataFrame:
        return self.db.dataframe(
            "SELECT t.ticket_id, t.subject, COALESCE(r.final_queue,d.destination_queue) final_queue, "
            "COALESCE(r.final_priority,p.priority_pred) final_priority, COALESCE(r.final_type,p.type_pred) final_type, "
            "n.normalization_status, n.subject_normalized, n.body_normalized, d.status, r.reviewer_decision, r.reviewer_notes, d.completed_at "
            "FROM tickets t JOIN normalizations n USING(ticket_id) JOIN predictions p USING(ticket_id) JOIN routing_decisions d USING(ticket_id) "
            "LEFT JOIN review_actions r ON r.id=(SELECT MAX(id) FROM review_actions WHERE ticket_id=t.ticket_id) WHERE d.status IN ('approved','rejected') ORDER BY t.ticket_id"
        )

    def jira_payload(self, ticket_id: str) -> dict[str, Any]:
        rows = self.export_actions()
        match = rows[rows["ticket_id"] == ticket_id]
        if match.empty:
            return {}
        row = match.iloc[0]
        return {"fields": {"externalId": row.ticket_id, "summary": row.subject, "project": {"key": "SUP"}, "issueType": {"name": row.final_type}, "priority": {"name": row.final_priority}, "labels": ["ai-governed"], "routingQueue": row.final_queue}, "lineage": {"source": "AI Workflow Control Plane", "ticketId": row.ticket_id}}
