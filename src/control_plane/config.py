from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AppConfig:
    model_version: str = "tfidf-sgd-v1"
    review_threshold: float = 0.45
    warning_threshold: float = 0.65
    high_priority_requires_review: bool = True
    baseline_change_review: bool = False
    risk_keywords: list[str] = field(
        default_factory=lambda: ["breach", "fraud", "locked out", "unauthorized", "lawsuit"]
    )
    restricted_tags: list[str] = field(
        default_factory=lambda: ["security", "billing dispute", "account access"]
    )
    supported_languages: list[str] = field(default_factory=lambda: ["en", "de"])
    language_queue: str = "Language Specialists"
    default_reviewer: str = "Support Lead"
    evaluation_mode: bool = True
    jira_payload_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AppConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})


DEFAULT_CONFIG = AppConfig()
