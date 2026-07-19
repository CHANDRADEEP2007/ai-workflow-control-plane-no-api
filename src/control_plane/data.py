from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

import pandas as pd


CANONICAL_COLUMNS = [
    "ticket_id",
    "subject",
    "body",
    "language",
    "queue_actual",
    "priority_actual",
    "type_actual",
    "business_type",
    "tags",
    "created_at",
]
REQUIRED_COLUMNS = {"ticket_id", "subject", "body"}
ALIASES = {
    "id": "ticket_id",
    "ticket id": "ticket_id",
    "ticketid": "ticket_id",
    "title": "subject",
    "ticket subject": "subject",
    "description": "body",
    "text": "body",
    "ticket body": "body",
    "lang": "language",
    "queue": "queue_actual",
    "category": "queue_actual",
    "priority": "priority_actual",
    "type": "type_actual",
    "ticket type": "type_actual",
    "business": "business_type",
    "industry": "business_type",
    "tag": "tags",
    "created": "created_at",
    "date": "created_at",
}


@dataclass(slots=True)
class DataQuality:
    rows: int
    valid_rows: int
    duplicate_ids: int
    missing_required: dict[str, int]
    optional_coverage: float
    issues: list[str]

    @property
    def is_valid(self) -> bool:
        return self.valid_rows > 0 and not any(v == self.rows for v in self.missing_required.values())


def clean_column_name(value: str) -> str:
    normalized = re.sub(r"[_\-]+", " ", str(value).strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    return ALIASES.get(normalized, normalized.replace(" ", "_"))


def infer_mapping(columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for source in columns:
        canonical = clean_column_name(source)
        if canonical in CANONICAL_COLUMNS and canonical not in mapping.values():
            mapping[source] = canonical
    return mapping


def read_csv(source: str | Path | IO[Any]) -> pd.DataFrame:
    return pd.read_csv(source, keep_default_na=False)


def normalize_dataframe(frame: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    mapping = dict(mapping or infer_mapping([str(c) for c in frame.columns]))
    tag_sources = [
        column for column in frame.columns
        if mapping.get(column) == "tags" or re.fullmatch(r"tags?(?:_?\d+)?", clean_column_name(column))
    ]
    combined_tags = None
    if tag_sources:
        combined_tags = frame[tag_sources].fillna("").astype(str).apply(
            lambda row: ", ".join(dict.fromkeys(value.strip() for value in row if value.strip())), axis=1
        )
        mapping = {source: target for source, target in mapping.items() if target != "tags"}
    normalized = frame.rename(columns=mapping).copy()
    if combined_tags is not None:
        normalized["tags"] = combined_tags
    for column in CANONICAL_COLUMNS:
        if column not in normalized:
            normalized[column] = ""
    normalized = normalized[CANONICAL_COLUMNS]
    for column in CANONICAL_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()
    missing_ids = normalized["ticket_id"].eq("")
    if missing_ids.any():
        normalized.loc[missing_ids, "ticket_id"] = [f"GEN-{i + 1:05d}" for i in range(missing_ids.sum())]
    normalized["language"] = normalized["language"].str.lower()
    priority = normalized["priority_actual"].str.lower()
    normalized.loc[priority.str.contains(r"(?:^|\D)1(?:\D|$)|low", regex=True), "priority_actual"] = "Low"
    normalized.loc[priority.str.contains(r"(?:^|\D)2(?:\D|$)|medium", regex=True), "priority_actual"] = "Medium"
    normalized.loc[priority.str.contains(r"(?:^|\D)3(?:\D|$)|high|critical", regex=True), "priority_actual"] = "High"
    normalized["created_at"] = normalized["created_at"].replace(
        "", datetime.now(timezone.utc).isoformat()
    )
    return normalized.drop_duplicates("ticket_id", keep="last").reset_index(drop=True)


def assess_quality(frame: pd.DataFrame, mapping: dict[str, str] | None = None) -> DataQuality:
    normalized = normalize_dataframe(frame, mapping)
    rows = len(frame)
    missing = {column: int(normalized[column].eq("").sum()) for column in REQUIRED_COLUMNS}
    duplicate_ids = int(frame.rename(columns=mapping or infer_mapping(list(frame.columns))).get("ticket_id", pd.Series(dtype=str)).duplicated().sum())
    optional = [column for column in CANONICAL_COLUMNS if column not in REQUIRED_COLUMNS]
    coverage = 0.0 if not rows else float((normalized[optional] != "").sum().sum() / (len(optional) * len(normalized)))
    issues = []
    if duplicate_ids:
        issues.append(f"{duplicate_ids} duplicate ticket ID(s) will be de-duplicated.")
    for column, count in missing.items():
        if count:
            issues.append(f"{count} row(s) are missing {column}.")
    if not issues:
        issues.append("No blocking data-quality issues detected.")
    valid_rows = int((normalized[list(REQUIRED_COLUMNS)] != "").all(axis=1).sum())
    return DataQuality(rows, valid_rows, duplicate_ids, missing, coverage, issues)
