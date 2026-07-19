from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .config import AppConfig, DEFAULT_CONFIG


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path = "data/control_plane.db") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY, subject TEXT NOT NULL, body TEXT NOT NULL,
                    language TEXT, queue_actual TEXT, priority_actual TEXT, type_actual TEXT,
                    business_type TEXT, tags TEXT, created_at TEXT, ingested_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    ticket_id TEXT PRIMARY KEY REFERENCES tickets(ticket_id) ON DELETE CASCADE,
                    queue_pred TEXT, priority_pred TEXT, type_pred TEXT,
                    confidence_queue REAL, confidence_priority REAL, confidence_type REAL,
                    explanation_text TEXT, model_version TEXT, predicted_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS normalizations (
                    ticket_id TEXT PRIMARY KEY REFERENCES tickets(ticket_id) ON DELETE CASCADE,
                    subject_normalized TEXT NOT NULL, body_normalized TEXT NOT NULL,
                    language_source TEXT NOT NULL, normalization_method TEXT NOT NULL,
                    normalization_status TEXT NOT NULL, normalization_confidence REAL NOT NULL,
                    fallback_used INTEGER NOT NULL, translation_model_version TEXT NOT NULL,
                    normalized_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS routing_decisions (
                    ticket_id TEXT PRIMARY KEY REFERENCES tickets(ticket_id) ON DELETE CASCADE,
                    final_action TEXT NOT NULL, destination_queue TEXT, rationale TEXT,
                    status TEXT NOT NULL, decided_at TEXT NOT NULL, completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS policy_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT REFERENCES tickets(ticket_id) ON DELETE CASCADE,
                    rule_id TEXT, rule_name TEXT, rule_result TEXT,
                    threshold_snapshot TEXT, event_time TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT REFERENCES tickets(ticket_id) ON DELETE CASCADE,
                    assigned_reviewer TEXT, reviewer_decision TEXT, reviewer_notes TEXT,
                    final_queue TEXT, final_priority TEXT, final_type TEXT, reviewed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT, stage TEXT NOT NULL, actor TEXT NOT NULL,
                    action TEXT NOT NULL, payload TEXT NOT NULL, event_time TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_policy_events_ticket ON policy_events(ticket_id);
                CREATE INDEX IF NOT EXISTS idx_normalizations_status ON normalizations(normalization_status);
                CREATE INDEX IF NOT EXISTS idx_normalizations_language ON normalizations(language_source);
                CREATE INDEX IF NOT EXISTS idx_review_actions_ticket ON review_actions(ticket_id);
                CREATE INDEX IF NOT EXISTS idx_audit_log_ticket ON audit_log(ticket_id);
                CREATE INDEX IF NOT EXISTS idx_audit_log_stage ON audit_log(stage);
                CREATE INDEX IF NOT EXISTS idx_routing_status ON routing_decisions(status);
                """
            )
            con.execute(
                "INSERT OR IGNORE INTO app_config(id, payload, updated_at) VALUES (1, ?, ?)",
                (json.dumps(DEFAULT_CONFIG.to_dict()), utcnow()),
            )

    def query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as con:
            return [dict(row) for row in con.execute(sql, parameters).fetchall()]

    def dataframe(self, sql: str, parameters: tuple[Any, ...] = ()) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query(sql, con, params=parameters)

    def count(self, table: str) -> int:
        allowed = {"tickets", "normalizations", "predictions", "routing_decisions", "policy_events", "review_actions", "audit_log"}
        if table not in allowed:
            raise ValueError("Unknown table")
        return int(self.query(f"SELECT COUNT(*) count FROM {table}")[0]["count"])

    def get_config(self) -> AppConfig:
        payload = self.query("SELECT payload FROM app_config WHERE id = 1")[0]["payload"]
        return AppConfig.from_dict(json.loads(payload))

    def save_config(self, config: AppConfig) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO app_config(id, payload, updated_at) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (json.dumps(config.to_dict()), utcnow()),
            )
        self.audit(None, "configuration", "admin", "config_updated", config.to_dict())

    def get_state(self, key: str, default: str = "") -> str:
        rows = self.query("SELECT value FROM app_state WHERE key=?", (key,))
        return str(rows[0]["value"]) if rows else default

    def set_state(self, key: str, value: str) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO app_state(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, utcnow()),
            )

    def upsert_tickets(self, frame: pd.DataFrame) -> int:
        now = utcnow()
        rows = [tuple(row[column] for column in frame.columns) + (now,) for _, row in frame.iterrows()]
        columns = list(frame.columns) + ["ingested_at"]
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns if column != "ticket_id")
        with self.connect() as con:
            con.executemany(
                f"INSERT INTO tickets({','.join(columns)}) VALUES({placeholders}) "
                f"ON CONFLICT(ticket_id) DO UPDATE SET {updates}", rows
            )
            con.executemany(
                "INSERT INTO audit_log(ticket_id, stage, actor, action, payload, event_time) VALUES(?,?,?,?,?,?)",
                [(str(ticket_id), "ingestion", "system", "ticket_normalized", '{"source": "csv"}', now) for ticket_id in frame["ticket_id"]],
            )
        return len(rows)

    def save_processing_batch(self, records: list[tuple[dict[str, Any], dict[str, Any], Any]], config: AppConfig) -> None:
        """Persist predictions, policy decisions, rule events, and audit events in one transaction."""
        if not records:
            return
        now = utcnow()
        prediction_rows = []
        decision_rows = []
        policy_rows = []
        audit_rows = []
        ticket_ids = []
        snapshot = json.dumps({"review_threshold": config.review_threshold, "warning_threshold": config.warning_threshold})
        for ticket, prediction, policy in records:
            ticket_id = ticket["ticket_id"]
            ticket_ids.append((ticket_id,))
            prediction_rows.append((
                ticket_id, prediction["queue_pred"], prediction["priority_pred"], prediction["type_pred"],
                prediction["confidence_queue"], prediction["confidence_priority"], prediction["confidence_type"],
                prediction["explanation_text"], prediction["model_version"], now,
            ))
            status = "pending_review" if policy.final_action == "human_review" else "approved"
            decision_rows.append((ticket_id, policy.final_action, policy.destination_queue, policy.rationale, status, now))
            policy_rows.extend(
                (ticket_id, rule["rule_id"], rule["name"], json.dumps(rule), snapshot, now)
                for rule in policy.triggered_rules
            )
            audit_rows.extend([
                (ticket_id, "prediction", "model", "recommendation_created", json.dumps(prediction), now),
                (ticket_id, "policy", "policy_engine", policy.final_action, json.dumps({"destination_queue": policy.destination_queue, "rules": policy.triggered_rules, "rationale": policy.rationale}), now),
            ])
        with self.connect() as con:
            con.executemany(
                "INSERT INTO predictions(ticket_id,queue_pred,priority_pred,type_pred,confidence_queue,confidence_priority,confidence_type,explanation_text,model_version,predicted_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(ticket_id) DO UPDATE SET queue_pred=excluded.queue_pred, priority_pred=excluded.priority_pred, "
                "type_pred=excluded.type_pred, confidence_queue=excluded.confidence_queue, confidence_priority=excluded.confidence_priority, "
                "confidence_type=excluded.confidence_type, explanation_text=excluded.explanation_text, model_version=excluded.model_version, predicted_at=excluded.predicted_at",
                prediction_rows,
            )
            con.executemany(
                "INSERT INTO routing_decisions(ticket_id,final_action,destination_queue,rationale,status,decided_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(ticket_id) DO UPDATE SET final_action=excluded.final_action, destination_queue=excluded.destination_queue, "
                "rationale=excluded.rationale, status=excluded.status, decided_at=excluded.decided_at, completed_at=NULL",
                decision_rows,
            )
            con.executemany("DELETE FROM policy_events WHERE ticket_id=?", ticket_ids)
            if policy_rows:
                con.executemany(
                    "INSERT INTO policy_events(ticket_id,rule_id,rule_name,rule_result,threshold_snapshot,event_time) VALUES(?,?,?,?,?,?)",
                    policy_rows,
                )
            con.executemany(
                "INSERT INTO audit_log(ticket_id,stage,actor,action,payload,event_time) VALUES(?,?,?,?,?,?)",
                audit_rows,
            )

    def save_normalizations(self, records: list[tuple[str, dict[str, Any]]]) -> None:
        if not records:
            return
        rows = []
        audits = []
        for ticket_id, result in records:
            rows.append((
                ticket_id, result["subject_normalized"], result["body_normalized"], result["language_source"],
                result["normalization_method"], result["normalization_status"], result["normalization_confidence"],
                int(result["fallback_used"]), result["translation_model_version"], result["normalized_at"],
            ))
            audits.append((ticket_id, "normalization", "local_normalizer", result["normalization_status"], json.dumps(result), result["normalized_at"]))
        with self.connect() as con:
            con.executemany(
                "INSERT INTO normalizations(ticket_id,subject_normalized,body_normalized,language_source,normalization_method,normalization_status,normalization_confidence,fallback_used,translation_model_version,normalized_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(ticket_id) DO UPDATE SET subject_normalized=excluded.subject_normalized, "
                "body_normalized=excluded.body_normalized, language_source=excluded.language_source, normalization_method=excluded.normalization_method, "
                "normalization_status=excluded.normalization_status, normalization_confidence=excluded.normalization_confidence, fallback_used=excluded.fallback_used, "
                "translation_model_version=excluded.translation_model_version, normalized_at=excluded.normalized_at",
                rows,
            )
            con.executemany(
                "INSERT INTO audit_log(ticket_id,stage,actor,action,payload,event_time) VALUES(?,?,?,?,?,?)",
                audits,
            )

    def save_prediction(self, ticket_id: str, prediction: dict[str, Any]) -> None:
        columns = list(prediction) + ["predicted_at"]
        values = [prediction[column] for column in prediction] + [utcnow()]
        updates = ",".join(f"{column}=excluded.{column}" for column in columns)
        with self.connect() as con:
            con.execute(
                f"INSERT INTO predictions(ticket_id,{','.join(columns)}) VALUES(?,{','.join('?' for _ in columns)}) "
                f"ON CONFLICT(ticket_id) DO UPDATE SET {updates}", [ticket_id] + values
            )
        self.audit(ticket_id, "prediction", "model", "recommendation_created", prediction)

    def save_decision(self, ticket_id: str, action: str, destination: str, rationale: str, rules: list[dict[str, Any]], threshold_snapshot: dict[str, Any]) -> None:
        now = utcnow()
        status = "pending_review" if action == "human_review" else "approved"
        with self.connect() as con:
            con.execute(
                "INSERT INTO routing_decisions(ticket_id, final_action, destination_queue, rationale, status, decided_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(ticket_id) DO UPDATE SET final_action=excluded.final_action, "
                "destination_queue=excluded.destination_queue, rationale=excluded.rationale, status=excluded.status, decided_at=excluded.decided_at, completed_at=NULL",
                (ticket_id, action, destination, rationale, status, now),
            )
            con.execute("DELETE FROM policy_events WHERE ticket_id = ?", (ticket_id,))
            con.executemany(
                "INSERT INTO policy_events(ticket_id, rule_id, rule_name, rule_result, threshold_snapshot, event_time) VALUES(?,?,?,?,?,?)",
                [(ticket_id, rule["rule_id"], rule["name"], json.dumps(rule), json.dumps(threshold_snapshot), now) for rule in rules],
            )
        self.audit(ticket_id, "policy", "policy_engine", action, {"destination_queue": destination, "rules": rules, "rationale": rationale})

    def save_review(self, ticket_id: str, reviewer: str, decision: str, notes: str, final_queue: str, final_priority: str, final_type: str) -> None:
        now = utcnow()
        with self.connect() as con:
            con.execute(
                "INSERT INTO review_actions(ticket_id, assigned_reviewer, reviewer_decision, reviewer_notes, final_queue, final_priority, final_type, reviewed_at) VALUES(?,?,?,?,?,?,?,?)",
                (ticket_id, reviewer, decision, notes, final_queue, final_priority, final_type, now),
            )
            con.execute(
                "UPDATE routing_decisions SET status=?, destination_queue=?, completed_at=? WHERE ticket_id=?",
                ("approved" if decision in {"approve", "reroute"} else "rejected", final_queue, now, ticket_id),
            )
        self.audit(ticket_id, "review", reviewer, decision, {"notes": notes, "final_queue": final_queue, "final_priority": final_priority, "final_type": final_type})

    def audit(self, ticket_id: str | None, stage: str, actor: str, action: str, payload: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO audit_log(ticket_id, stage, actor, action, payload, event_time) VALUES(?,?,?,?,?,?)",
                (ticket_id, stage, actor, action, json.dumps(payload, default=str), utcnow()),
            )

    def reset(self, reset_setup: bool = False) -> None:
        with self.connect() as con:
            for table in ("review_actions", "policy_events", "routing_decisions", "predictions", "normalizations", "tickets", "audit_log"):
                con.execute(f"DELETE FROM {table}")
            if reset_setup:
                con.execute("DELETE FROM app_state WHERE key IN ('setup_started','setup_complete','policy_confirmed')")
