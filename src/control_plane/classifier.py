from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier


@dataclass(slots=True)
class ModelBundle:
    vectorizer: TfidfVectorizer
    models: dict[str, Any]
    version: str = "tfidf-sgd-v1"


def _text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(field, ""))
        for field in ("subject_normalized", "body_normalized", "tags", "language_source")
    )


class LocalTicketClassifier:
    """Credential-free TF-IDF classifiers with probabilistic linear outputs."""

    targets = {"queue": "queue_actual", "priority": "priority_actual", "type": "type_actual"}

    def __init__(self, version: str = "tfidf-sgd-v1") -> None:
        self.version = version
        self.bundle: ModelBundle | None = None

    def fit(self, rows: list[dict[str, Any]]) -> None:
        training_rows = rows if len(rows) < 10 else [row for row in rows if not is_evaluation_holdout(str(row.get("ticket_id", "")))]
        texts = [_text(row) for row in training_rows]
        small_dataset = len(training_rows) < 10
        vectorizer = TfidfVectorizer(
            max_features=18000, ngram_range=(1, 2), min_df=1 if small_dataset else 2, max_df=1.0 if small_dataset else 0.995,
            sublinear_tf=True, strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(texts)
        models: dict[str, Any] = {}
        for name, field in self.targets.items():
            labels = [str(row.get(field, "") or "Unknown") for row in training_rows]
            if len(set(labels)) == 1:
                model = DummyClassifier(strategy="most_frequent")
            else:
                model = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=100, tol=1e-3, random_state=42, class_weight="balanced")
            model.fit(matrix, labels)
            models[name] = model
        self.bundle = ModelBundle(vectorizer, models, self.version)

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.bundle:
            raise RuntimeError("Classifier must be fitted before prediction")
        matrix = self.bundle.vectorizer.transform([_text(row) for row in rows])
        outputs = [dict() for _ in rows]
        feature_names = self.bundle.vectorizer.get_feature_names_out()
        for name, model in self.bundle.models.items():
            probabilities = model.predict_proba(matrix)
            best = np.argmax(probabilities, axis=1)
            for index, class_index in enumerate(best):
                outputs[index][f"{name}_pred"] = str(model.classes_[class_index])
                outputs[index][f"confidence_{name}"] = round(float(probabilities[index, class_index]), 3)
        for index, row in enumerate(rows):
            vector = matrix[index]
            if vector.nnz:
                top_positions = vector.indices[np.argsort(vector.data)[-4:]][::-1]
                signals = ", ".join(feature_names[top_positions])
            else:
                signals = "metadata fallback"
            outputs[index]["explanation_text"] = f"Local TF-IDF linear model; strongest normalized-text signals: {signals}."
            outputs[index]["model_version"] = self.version
        return outputs


def is_evaluation_holdout(ticket_id: str) -> bool:
    """Stable 20% split without persisting or exposing target labels to the model."""
    digest = hashlib.sha1(ticket_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 5 == 0
