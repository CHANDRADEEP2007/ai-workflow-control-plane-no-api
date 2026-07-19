# AI Workflow Control Plane — No-API MVP

An enterprise-style, credential-free Streamlit MVP for governed multilingual support-ticket triage. The application preserves original language, builds a local English-normalized layer, predicts queue, priority, and issue type with scikit-learn, applies policy safeguards, sends exceptions to human reviewers, and records every decision in SQLite.

## What is implemented

- CSV upload, inferred column mapping, quality checks, normalization, and a 30-ticket sample dataset
- Compatibility with the PRD's Kaggle dataset shape, including numeric priority labels and split Tag 1–Tag 10 columns
- Offline English normalization for English/German intake, with immutable original text and an explicit fallback status
- Local TF-IDF plus probabilistic linear classifiers with per-field confidence and human-readable signals
- Deterministic 20% holdout evaluation so displayed accuracy is measured on unseen tickets
- Configurable confidence thresholds, risk keywords, restricted tags, language routing, and evaluation safeguards
- Auto-route, auto-route-with-warning, and human-review decision paths
- Priority-sorted review queue with approve, reject, reroute, notes, and final-field overrides
- Per-ticket event lineage with raw, normalization, prediction, policy, and reviewer snapshots
- Accuracy, review, override, policy-hit, confidence, language mix, translation coverage/failure, throughput, and audit-completeness analytics
- CSV handoff plus Jira-ready payload simulation
- Persistent model/admin configuration and a polished portfolio case-study view

## Run locally

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

In the app, open **Dataset Ingestion**, keep **Provided multilingual dataset** selected, save the records, then open **Model Predictions** and score the batch. When the supplied download is not available, the app falls back to the bundled synthetic sample.

## Test

```powershell
pip install -r requirements-dev.txt
$env:PYTHONPATH = "src"
pytest -q
```

## Architecture

```text
Streamlit presentation
        │
        ▼
ControlPlane orchestrator
   ├── immutable source data
   ├── local language normalization
   ├── TF-IDF linear prediction + confidence
   ├── policy evaluation
   ├── reviewer workflow
   └── analytics/export
        │
        ▼
SQLite tickets, predictions, decisions, rules, reviews, config, and audit events
```

The classifier and normalizer are intentionally local and explainable. They can be replaced behind `src/control_plane/classifier.py` and `src/control_plane/normalizer.py` without changing the policy, review, or audit layers.

## Data and privacy

The bundled sample records are synthetic. Runtime state is stored in `data/control_plane.db`, which is ignored by Git. No external API keys, hosted models, or translation services are required.

For German tickets, the MVP uses a transparent local support-domain lexicon plus tag metadata and marks the result as `normalized_with_fallback`. Unsupported or failed language handling is preserved verbatim and forced to review by policy.

The PRD's source dataset is Tobias Bueck's [Customer IT Support - Ticket Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets), licensed CC BY 4.0. Download any CSV variant and upload it through **Dataset Ingestion**; the mapper handles its queue, priority, language, subject, body, type, business type, and split tag fields.
