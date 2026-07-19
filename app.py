from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from control_plane.config import AppConfig, DEFAULT_CONFIG
from control_plane.data import CANONICAL_COLUMNS, infer_mapping, read_csv
from control_plane.service import ControlPlane


st.set_page_config(page_title="AI Workflow Control Plane", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    :root { --ink:#10231f; --muted:#65736f; --accent:#0f766e; --mint:#dff7f1; --amber:#fff2cc; --red:#ffe3e3; }
    .stApp { background: #f6f8f7; color: var(--ink); }
    [data-testid="stSidebar"] { background: #10231f; }
    [data-testid="stSidebar"] * { color: #eef7f4; }
    [data-testid="stMetric"] { background:white; border:1px solid #dce5e2; border-radius:14px; padding:16px; box-shadow:0 4px 18px rgba(16,35,31,.05); }
    .hero { padding:34px; border-radius:24px; color:white; background:linear-gradient(135deg,#10231f,#0f766e); margin-bottom:24px; }
    .hero h1 { font-size:44px; line-height:1.05; margin:0 0 12px; }
    .hero p { max-width:780px; font-size:18px; color:#d7ebe6; }
    .eyebrow { text-transform:uppercase; letter-spacing:.12em; font-weight:700; color:#8de0cf; font-size:12px; }
    .panel { background:white; border:1px solid #dce5e2; border-radius:14px; padding:18px; margin:8px 0 16px; }
    .decision { border-left:6px solid #0f766e; background:white; border-radius:12px; padding:18px; }
    .muted { color:#65736f; }
    .pill { display:inline-block; padding:5px 10px; border-radius:99px; margin:2px; background:#dff7f1; color:#0f615b; font-size:12px; font-weight:700; }
    .timeline { border-left:2px solid #b9d4ce; margin-left:8px; padding-left:18px; padding-bottom:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_plane() -> ControlPlane:
    return ControlPlane(ROOT / "data" / "control_plane.db")


plane = get_plane()

PROVIDED_DATASET = Path.home() / "Downloads" / "aa_dataset-tickets-multi-lang-5-2-50-version (1).csv"
DEFAULT_DATASET = PROVIDED_DATASET if PROVIDED_DATASET.exists() else ROOT / "data" / "sample_tickets.csv"

SCREENS = [
    "Overview", "Dataset Ingestion", "Ticket Explorer", "Model Predictions",
    "Policy Engine", "Routing Decisions", "Human Review Queue", "Audit Trail",
    "Analytics", "Export", "Admin", "Portfolio Case Study",
]


def set_screen(name: str) -> None:
    st.session_state.screen = name


def moneyless_percent(value: float) -> str:
    return f"{value:.1%}"


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def ticket_selector(data: pd.DataFrame, key: str, label: str = "Ticket") -> str | None:
    if data.empty:
        st.info("No tickets are available yet. Load and process a dataset first.")
        return None
    labels = {row.ticket_id: f"{row.ticket_id} — {row.subject[:70]}" for row in data.itertuples()}
    selected = st.selectbox(label, list(labels), format_func=labels.get, key=key)
    st.session_state.selected_ticket = selected
    return selected


def overview() -> None:
    metrics = plane.overview()
    st.markdown(
        """<section class="hero"><div class="eyebrow">AI Operations · Governed by design</div>
        <h1>Govern multilingual support routing.<br>No API credentials required.</h1>
        <p>Normalize multilingual intake locally, classify with explainable machine learning, apply policy safeguards, and preserve human control with complete lineage.</p></section>""",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Tickets", metrics["tickets"])
    cols[1].metric("Processed", metrics["processed"])
    cols[2].metric("Awaiting review", metrics["pending_review"])
    cols[3].metric("Auto-route rate", moneyless_percent(metrics["auto_route_rate"]))
    st.subheader("One governed decision pipeline")
    steps = [
        ("01", "Ingest", "Validate and normalize support data."),
        ("02", "Normalize", "Create an English layer while preserving original text."),
        ("03", "Predict", "Run local TF-IDF classification with confidence."),
        ("04", "Govern", "Apply language, confidence, and risk policies."),
        ("05", "Review", "Resolve exceptions and retain full lineage."),
    ]
    for col, (number, name, detail) in zip(st.columns(5), steps):
        col.markdown(f"<div class='panel'><span class='eyebrow' style='color:#0f766e'>{number}</span><h3>{name}</h3><p class='muted'>{detail}</p></div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    if a.button("Load dataset", type="primary", width="stretch"):
        set_screen("Dataset Ingestion"); st.rerun()
    if b.button("Open review queue", width="stretch"):
        set_screen("Human Review Queue"); st.rerun()
    if c.button("Inspect policies", width="stretch"):
        set_screen("Policy Engine"); st.rerun()


def ingestion() -> None:
    header("Dataset Ingestion", "Upload a CSV, map its schema, validate quality, and save normalized records.")
    default_label = "Provided multilingual dataset" if PROVIDED_DATASET.exists() else "Bundled sample dataset"
    source_choice = st.radio("Source", [default_label, "Upload CSV"], horizontal=True)
    st.caption("The provided Customer IT Support dataset is the default when available. Its split Tag 1–Tag 8 columns are merged during normalization; records missing required subjects remain visible in the quality summary and are not saved.")
    source = DEFAULT_DATASET if source_choice == default_label else st.file_uploader("Choose a CSV", type=["csv"])
    if source is None:
        st.info("Choose a CSV to continue."); return
    try:
        raw = read_csv(source)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}"); return
    st.subheader("Column mapping")
    inferred = infer_mapping(list(raw.columns))
    mapping: dict[str, str] = {}
    options = ["Ignore"] + CANONICAL_COLUMNS
    map_cols = st.columns(3)
    for index, column in enumerate(raw.columns):
        default = inferred.get(column, "Ignore")
        selected = map_cols[index % 3].selectbox(str(column), options, index=options.index(default), key=f"map_{column}")
        if selected != "Ignore": mapping[column] = selected
    from control_plane.data import assess_quality
    quality = assess_quality(raw, mapping)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Source rows", quality.rows)
    q2.metric("Valid rows", quality.valid_rows)
    q3.metric("Duplicate IDs", quality.duplicate_ids)
    q4.metric("Optional coverage", moneyless_percent(quality.optional_coverage))
    for issue in quality.issues:
        st.caption(f"• {issue}")
    language_source = next((column for column, target in mapping.items() if target == "language"), None)
    if language_source:
        language_counts = raw[language_source].fillna("unknown").astype(str).str.lower().value_counts().rename_axis("language").reset_index(name="tickets")
        st.plotly_chart(px.bar(language_counts, x="language", y="tickets", color="language", title="Source language distribution"), width="stretch")
    st.dataframe(raw.head(25), width="stretch", hide_index=True)
    if st.button("Validate and save records", type="primary", disabled=not quality.is_valid):
        count, _ = plane.ingest(raw, mapping)
        normalized = plane.normalize_pending()
        st.success(f"Saved {count} tickets and created {normalized} local normalization records.")
    if plane.db.count("tickets") and st.button("Continue to processing"):
        set_screen("Model Predictions"); st.rerun()


def explorer() -> None:
    header("Ticket Explorer", "Search, filter, and inspect normalized inputs before or after processing.")
    data = plane.ticket_view()
    if data.empty:
        st.info("Load a dataset to populate the explorer."); return
    search = st.text_input("Search subject, body, ID, or tags", placeholder="Try: invoice, locked out, security...")
    c1, c2, c3, c4 = st.columns(4)
    filters = {
        "language": c1.multiselect("Language", sorted(data.language.dropna().unique())),
        "queue_actual": c2.multiselect("Actual queue", sorted(data.queue_actual.dropna().unique())),
        "priority_actual": c3.multiselect("Priority", sorted(data.priority_actual.dropna().unique())),
        "type_actual": c4.multiselect("Type", sorted(data.type_actual.dropna().unique())),
    }
    shown = data.copy()
    for column, selected in filters.items():
        if selected: shown = shown[shown[column].isin(selected)]
    if search:
        mask = shown[["ticket_id", "subject", "body", "tags"]].fillna("").apply(lambda col: col.str.contains(search, case=False, regex=False)).any(axis=1)
        shown = shown[mask]
    st.dataframe(shown[["ticket_id", "subject", "language", "normalization_status", "queue_actual", "priority_actual", "type_actual", "tags"]], width="stretch", hide_index=True)
    selected = ticket_selector(shown, "explorer_ticket", "Open ticket detail")
    if selected:
        row = shown[shown.ticket_id == selected].iloc[0]
        original, normalized = st.columns(2)
        with original, st.container(border=True):
            st.caption(f"ORIGINAL · {str(row.language_source or row.language).upper()}")
            st.subheader(row.subject); st.write(row.body)
        with normalized, st.container(border=True):
            st.caption("ENGLISH-NORMALIZED")
            st.subheader(row.subject_normalized or "Not normalized"); st.write(row.body_normalized or "Run local processing to create normalized text.")
        st.caption(f"Method: {row.normalization_method or 'pending'} · Status: {row.normalization_status or 'pending'} · Confidence: {float(row.normalization_confidence or 0):.0%}")
        st.markdown(" ".join(f"<span class='pill'>{tag.strip()}</span>" for tag in str(row.tags).split(",") if tag.strip()), unsafe_allow_html=True)


def predictions() -> None:
    header("Model Predictions", "Inspect credential-free TF-IDF recommendations based on locally normalized text.")
    total = plane.db.count("tickets")
    if not total:
        st.info("Load records before running predictions."); return
    if st.button(f"Score all {total} tickets", type="primary"):
        processed = plane.process(); st.success(f"Scored and governed {processed} tickets.")
    data = plane.ticket_view()
    scored = data[data.queue_pred.notna()].copy()
    if scored.empty: return
    sort_field = st.selectbox("Sort by", ["Lowest overall confidence", "Queue confidence", "Priority confidence", "Type confidence"])
    scored["min_confidence"] = scored[["confidence_queue", "confidence_priority", "confidence_type"]].min(axis=1)
    sort_column = {"Lowest overall confidence": "min_confidence", "Queue confidence": "confidence_queue", "Priority confidence": "confidence_priority", "Type confidence": "confidence_type"}[sort_field]
    scored = scored.sort_values(sort_column)
    st.dataframe(scored[["ticket_id", "subject", "queue_pred", "confidence_queue", "priority_pred", "confidence_priority", "type_pred", "confidence_type"]], width="stretch", hide_index=True, column_config={c: st.column_config.ProgressColumn(c.replace("confidence_", "").title(), min_value=0, max_value=1, format="%.0%%") for c in ["confidence_queue", "confidence_priority", "confidence_type"]})
    selected = ticket_selector(scored, "prediction_ticket", "Per-ticket reasoning")
    if selected:
        row = scored[scored.ticket_id == selected].iloc[0]
        cols = st.columns(3)
        for col, field, actual in zip(cols, ["queue", "priority", "type"], ["queue_actual", "priority_actual", "type_actual"]):
            col.metric(field.title(), row[f"{field}_pred"], f"Actual: {row[actual]}" if row[actual] else None)
            col.progress(float(row[f"confidence_{field}"]), text=f"{row[f'confidence_{field}']:.0%} confidence")
        st.info(row.explanation_text)


def policy_console() -> None:
    header("Policy Engine Console", "Change governance rules and immediately see downstream routing outcomes.")
    config = plane.config
    with st.form("policy_form"):
        c1, c2 = st.columns(2)
        review = c1.slider("Human review below", 0.3, 0.9, config.review_threshold, 0.01)
        warning = c2.slider("Warning below", review, 0.98, max(review, config.warning_threshold), 0.01)
        risk = st.text_area("High-risk keywords (comma separated)", ", ".join(config.risk_keywords))
        tags = st.text_area("Restricted tags (comma separated)", ", ".join(config.restricted_tags))
        c3, c4 = st.columns(2)
        high = c3.toggle("Review moderate-confidence high-priority cases", config.high_priority_requires_review)
        baseline = c4.toggle("Review baseline queue disagreements", config.baseline_change_review)
        save = st.form_submit_button("Save policy and re-evaluate", type="primary")
    if save:
        config.review_threshold = review; config.warning_threshold = warning
        config.risk_keywords = [item.strip() for item in risk.split(",") if item.strip()]
        config.restricted_tags = [item.strip() for item in tags.split(",") if item.strip()]
        config.high_priority_requires_review = high; config.baseline_change_review = baseline
        plane.save_config(config); st.success("Policy saved and all tickets re-evaluated.")
    st.subheader("Active rules")
    rules = [
        ("NORMALIZATION_FAILED", "Normalization failure", "Unsupported or failed language handling forces review", True),
        ("NORMALIZATION_FALLBACK", "Normalization fallback", "High-priority fallback cases force review; others warn", True),
        ("CONF_LOW", "Low confidence", f"Review when any prediction is below {config.review_threshold:.0%}", True),
        ("CONF_WARN", "Moderate confidence", f"Warn when any prediction is below {config.warning_threshold:.0%}", True),
        ("TAG_RISK", "Restricted tag", ", ".join(config.restricted_tags), bool(config.restricted_tags)),
        ("KEYWORD_RISK", "Risk keyword", ", ".join(config.risk_keywords), bool(config.risk_keywords)),
        ("HIGH_PRIORITY", "Priority safeguard", "Review moderate-confidence High cases", config.high_priority_requires_review),
        ("BASELINE_CHANGE", "Baseline disagreement", "Review queue changes in evaluation mode", config.baseline_change_review),
    ]
    st.dataframe(pd.DataFrame(rules, columns=["Rule ID", "Rule", "Behavior", "Enabled"]), width="stretch", hide_index=True)
    decisions = plane.ticket_view()
    if not decisions.empty and decisions.final_action.notna().any():
        counts = decisions.final_action.value_counts().rename_axis("action").reset_index(name="tickets")
        st.plotly_chart(px.bar(counts, x="action", y="tickets", color="action", title="Simulated routing outcomes"), width="stretch")


def routing() -> None:
    header("Routing Decision View", "Trace the separation between model recommendation and policy-governed action.")
    data = plane.ticket_view().dropna(subset=["final_action"])
    selected = ticket_selector(data, "routing_ticket")
    if not selected: return
    trace = plane.ticket_trace(selected); ticket, normalization, pred, decision = trace["ticket"], trace["normalization"], trace["prediction"], trace["decision"]
    action_labels = {"auto_route": "Auto-route", "auto_route_warning": "Auto-route with warning", "human_review": "Human review required"}
    st.markdown(f"<div class='decision'><div class='eyebrow' style='color:#0f766e'>Final governed action</div><h2>{action_labels.get(decision.get('final_action'), decision.get('final_action'))}</h2><p>Destination: <strong>{decision.get('destination_queue')}</strong></p><p class='muted'>{decision.get('rationale')}</p></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Model recommendation")
        st.write(f"Queue: **{pred.get('queue_pred')}** ({pred.get('confidence_queue',0):.0%})")
        st.write(f"Priority: **{pred.get('priority_pred')}** ({pred.get('confidence_priority',0):.0%})")
        st.write(f"Type: **{pred.get('type_pred')}** ({pred.get('confidence_type',0):.0%})")
        st.caption(pred.get("explanation_text", ""))
    with c2:
        st.subheader("Triggered rules")
        if not trace["rules"]: st.success("No rules triggered.")
        for rule in trace["rules"]:
            result = json.loads(rule["rule_result"]); st.warning(f"**{rule['rule_name']}** — {result.get('detail')}")
    with st.expander("Original → normalized text used by the model"):
        left, right = st.columns(2)
        left.markdown(f"**Original ({ticket.get('language','unknown')})**\n\n{ticket.get('subject','')}\n\n{ticket.get('body','')}")
        right.markdown(f"**English-normalized**\n\n{normalization.get('subject_normalized','')}\n\n{normalization.get('body_normalized','')}")
    if decision.get("status") != "pending_review" and st.button("Send to reviewer anyway"):
        with plane.db.connect() as con: con.execute("UPDATE routing_decisions SET final_action='human_review', status='pending_review' WHERE ticket_id=?", (selected,))
        plane.db.audit(selected, "policy", "operator", "manual_review_requested", {})
        st.success("Added to the review queue."); st.rerun()


def review_queue() -> None:
    header("Human Review Queue", "Resolve policy exceptions while preserving full decision lineage.")
    queue = plane.review_queue()
    if queue.empty:
        st.success("No tickets are waiting for review."); return
    high = int((queue.priority_pred == "High").sum())
    c1, c2, c3 = st.columns(3); c1.metric("Awaiting review", len(queue)); c2.metric("High priority", high); c3.metric("Assigned reviewer", plane.config.default_reviewer)
    queue["SLA"] = queue.created_at.apply(lambda _: "Due soon")
    st.dataframe(queue[["ticket_id", "subject", "priority_pred", "destination_queue", "SLA"]], width="stretch", hide_index=True)
    selected = ticket_selector(queue, "review_ticket", "Review case")
    if not selected: return
    row = queue[queue.ticket_id == selected].iloc[0]
    left, right = st.columns([1.1, 1])
    with left:
        original, normalized = st.tabs([f"Original · {str(row.language_source).upper()}", "English-normalized"])
        with original:
            st.subheader(row.subject); st.write(row.body)
        with normalized:
            st.subheader(row.subject_normalized); st.write(row.body_normalized)
        st.caption(f"Method: {row.normalization_method} · Status: {row.normalization_status} · Confidence: {row.normalization_confidence:.0%}")
        st.caption(f"Tags: {row.tags or 'none'}")
        st.info(f"AI rationale: {row.explanation_text}\n\nPolicy rationale: {row.rationale}")
    with right, st.form("review_form"):
        decision = st.radio("Decision", ["approve", "reroute", "reject"], horizontal=True)
        reviewer = st.text_input("Reviewer", plane.config.default_reviewer)
        final_queue = st.text_input("Final queue", row.destination_queue)
        final_priority = st.selectbox("Final priority", ["Low", "Medium", "High"], index=["Low", "Medium", "High"].index(row.priority_pred))
        final_type = st.text_input("Final type", row.type_pred)
        notes = st.text_area("Reviewer notes", placeholder="Explain any override or rejection...")
        submitted = st.form_submit_button("Submit decision", type="primary")
    if submitted:
        plane.review(selected, decision, reviewer, notes, final_queue, final_priority, final_type)
        st.success("Reviewer decision stored with full traceability."); st.rerun()


def audit_trail() -> None:
    header("Audit Trail", "Search a ticket and inspect every recorded event, snapshot, and human action.")
    data = plane.ticket_view()
    selected = ticket_selector(data, "audit_ticket", "Search by ticket")
    if not selected: return
    trace = plane.ticket_trace(selected)
    download = pd.DataFrame(trace["audit"]).to_csv(index=False).encode()
    st.download_button("Export this audit history", download, f"audit-{selected}.csv", "text/csv")
    for event in trace["audit"]:
        with st.expander(f"{event['event_time']} · {event['stage'].title()} · {event['action']}", expanded=True):
            st.caption(f"Actor: {event['actor']}"); st.json(json.loads(event["payload"]))
    with st.expander("Raw ticket payload"): st.json(trace["ticket"])
    with st.expander("Prediction snapshot"): st.json(trace["prediction"])
    with st.expander("Normalization snapshot"): st.json(trace["normalization"])


def analytics() -> None:
    header("Analytics Dashboard", "Measure model quality, policy behavior, and operational outcomes together.")
    result = plane.analytics(); data = result["data"]
    if data.empty:
        st.info("Process tickets to populate analytics."); return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdout queue accuracy", moneyless_percent(result["queue_accuracy"]))
    c2.metric("Holdout priority accuracy", moneyless_percent(result["priority_accuracy"]))
    c3.metric("Review rate", moneyless_percent(result["review_rate"]))
    c4.metric("Override rate", moneyless_percent(result["override_rate"]))
    c5, c6, c7 = st.columns(3)
    c5.metric("Translation coverage", moneyless_percent(result["translation_coverage"]))
    c6.metric("Translation failure", moneyless_percent(result["translation_failure_rate"]))
    c7.metric("Multilingual handling time", f"{result['reviewer_handling_minutes']:.1f} min")
    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        queues = f1.multiselect("Queue", sorted(data.queue_pred.dropna().unique()))
        priorities = f2.multiselect("Priority", sorted(data.priority_pred.dropna().unique()))
        languages = f3.multiselect("Language", sorted(data.language.dropna().unique()))
    filtered = data
    if queues: filtered = filtered[filtered.queue_pred.isin(queues)]
    if priorities: filtered = filtered[filtered.priority_pred.isin(priorities)]
    if languages: filtered = filtered[filtered.language.isin(languages)]
    left, right = st.columns(2)
    actions = filtered.final_action.value_counts().rename_axis("action").reset_index(name="tickets")
    left.plotly_chart(px.pie(actions, names="action", values="tickets", hole=.58, title="Routing outcomes"), width="stretch")
    confidences = filtered[["confidence_queue", "confidence_priority", "confidence_type"]].melt(var_name="field", value_name="confidence")
    right.plotly_chart(px.histogram(confidences, x="confidence", color="field", nbins=12, barmode="overlay", title="Confidence distribution"), width="stretch")
    rule_data = result["rules"]
    if not rule_data.empty:
        hits = rule_data.rule_name.value_counts().rename_axis("rule").reset_index(name="hits")
        st.plotly_chart(px.bar(hits, x="hits", y="rule", orientation="h", title="Policy hit distribution"), width="stretch")
    q = filtered.queue_pred.value_counts().rename_axis("queue").reset_index(name="tickets")
    st.plotly_chart(px.bar(q, x="queue", y="tickets", color="queue", title="Queue routing volume"), width="stretch")
    lang, methods = st.columns(2)
    language_mix = filtered.language_source.fillna("unknown").value_counts().rename_axis("language").reset_index(name="tickets")
    lang.plotly_chart(px.pie(language_mix, names="language", values="tickets", hole=.5, title="Language mix"), width="stretch")
    method_mix = filtered.normalization_method.fillna("pending").value_counts().rename_axis("method").reset_index(name="tickets")
    methods.plotly_chart(px.bar(method_mix, x="tickets", y="method", orientation="h", title="Normalization methods"), width="stretch")
    st.caption(f"Type accuracy: {result['type_accuracy']:.1%} · Audit completeness: {result['audit_completeness']:.1%} · Reviewed throughput: {len(result['reviews'])} decisions")


def export_page() -> None:
    header("Export / Downstream Action", "Package final actions for handoff without breaking lineage.")
    actions = plane.export_actions()
    if actions.empty:
        st.info("No approved or rejected actions are ready for export."); return
    a, b, c = st.columns(3); a.metric("Ready records", len(actions)); b.metric("Approved", int((actions.status == "approved").sum())); c.metric("Rejected", int((actions.status == "rejected").sum()))
    st.dataframe(actions, width="stretch", hide_index=True)
    st.download_button("Download final actions CSV", actions.to_csv(index=False).encode(), "governed-support-actions.csv", "text/csv", type="primary")
    if plane.config.jira_payload_enabled:
        selected = st.selectbox("Jira-ready payload preview", actions.ticket_id.tolist())
        st.json(plane.jira_payload(selected))
    st.divider()
    if st.button("Reset demo state"):
        st.session_state.confirm_reset = True
    if st.session_state.get("confirm_reset"):
        st.warning("This permanently removes local demo tickets, predictions, reviews, and audit events.")
        if st.button("Confirm reset", type="primary"):
            plane.db.reset(); st.session_state.confirm_reset = False; st.success("Demo state reset."); st.rerun()


def admin() -> None:
    header("Admin / Configuration", "Centralize reusable settings for demo operation and policy iteration.")
    config = plane.config
    with st.form("admin_form"):
        c1, c2 = st.columns(2)
        model = c1.text_input("Model version", config.model_version)
        reviewer = c2.text_input("Default reviewer", config.default_reviewer)
        languages = st.text_input("Supported languages", ", ".join(config.supported_languages))
        language_queue = st.text_input("Language-specialist queue", config.language_queue)
        c3, c4 = st.columns(2)
        evaluation = c3.toggle("Evaluation mode", config.evaluation_mode)
        jira = c4.toggle("Jira payload preview", config.jira_payload_enabled)
        submitted = st.form_submit_button("Save configuration", type="primary")
    if submitted:
        config.model_version = model; config.default_reviewer = reviewer
        config.supported_languages = [item.strip().lower() for item in languages.split(",") if item.strip()]
        config.language_queue = language_queue; config.evaluation_mode = evaluation; config.jira_payload_enabled = jira
        plane.save_config(config); st.success("Configuration persisted and processing state refreshed.")
    if st.button("Restore defaults"):
        plane.save_config(AppConfig.from_dict(DEFAULT_CONFIG.to_dict())); st.success("Default configuration restored."); st.rerun()
    st.subheader("System status")
    status = plane.overview()
    st.json({"database": str(plane.db.path), "model": plane.config.model_version, "external_api_credentials_required": False, "supported_languages": plane.config.supported_languages, "tickets": status["tickets"], "audit_events": status["audit_events"], "timestamp_utc": datetime.now(timezone.utc).isoformat()})


def case_study() -> None:
    header("Portfolio Case Study", "A recruiter-friendly view of the product judgment behind the control plane.")
    st.markdown("""<section class="hero"><div class="eyebrow">Local-first · credential-free</div><h1>Govern multilingual AI-assisted support routing without external APIs.</h1><p>A production-minded workflow that preserves original language, creates a local normalized layer, and makes every automated boundary inspectable.</p></section>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("### Problem\nManual triage is slow, but unconstrained automation creates operational and compliance risk.")
    c2.markdown("### Product bet\nNormalize locally, treat model output as a workflow input, then add policy controls, exception handling, and lineage.")
    c3.markdown("### Outcome\nA clear operating model for safe auto-routing, rapid human review, and continuous governance learning.")
    st.subheader("Architecture")
    st.code("Streamlit UI → Local Normalizer → TF-IDF Classifiers → Policy Engine\n                      ↓               ↓                 ↓\n                 SQLite lineage ← Human Review ← Exceptions\n                      ↓\n              CSV / Jira-ready export", language="text")
    metrics = plane.analytics()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Queue accuracy", moneyless_percent(metrics["queue_accuracy"]))
    m2.metric("Review rate", moneyless_percent(metrics["review_rate"]))
    m3.metric("Override rate", moneyless_percent(metrics["override_rate"]))
    m4.metric("Audit completeness", moneyless_percent(metrics["audit_completeness"]))
    st.subheader("Current state → target state")
    current, target = st.columns(2)
    current.error("**Before**\n\nManual queue assignment · Hidden decision logic · Inconsistent escalation · Limited evidence")
    target.success("**With the control plane**\n\nGoverned auto-routing · Visible rationale · Human exception queue · End-to-end auditability")
    st.subheader("Key product tradeoffs")
    st.markdown("- A local TF-IDF linear pipeline keeps classification explainable and credential-free.\n- Original language is immutable; normalization is a separate, fully audited layer.\n- Confidence never acts alone; language failures and explicit safeguards represent operational risk.\n- Review friction is concentrated on exceptions, while routine cases remain fast.")


RENDERERS = {
    "Overview": overview, "Dataset Ingestion": ingestion, "Ticket Explorer": explorer,
    "Model Predictions": predictions, "Policy Engine": policy_console, "Routing Decisions": routing,
    "Human Review Queue": review_queue, "Audit Trail": audit_trail, "Analytics": analytics,
    "Export": export_page, "Admin": admin, "Portfolio Case Study": case_study,
}

with st.sidebar:
    st.markdown("## ◈ Control Plane")
    st.caption("Support Operations · MVP")
    current = st.session_state.get("screen", "Overview")
    selected_screen = st.radio("Navigate", SCREENS, index=SCREENS.index(current), label_visibility="collapsed")
    st.session_state.screen = selected_screen
    st.divider()
    status = plane.overview()
    st.caption(f"{status['tickets']} tickets · {status['pending_review']} awaiting review")
    st.caption(f"Model: {plane.config.model_version}")

RENDERERS[st.session_state.screen]()
