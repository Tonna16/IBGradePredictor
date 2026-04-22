import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import streamlit as st
from fpdf import FPDF

from core.io import subject_from_dict, subject_to_dict
from core.predictor import (
    GRADE_BOUNDS,
    Subject,
    SubjectPrediction,
    exam_component_from_trajectory,
    grade_from_score,
    needed_ia_quality_for_target,
    normalize_weights,
    parse_iso_date,
    predict,
    raw_to_pct,
    required_remaining_exam_average,
    risk_alerts,
)
from ml.forecaster import (
    ArtifactLoadStatus,
    build_features,
    load_model_bundle,
    predict_with_bundle,
    train_models_from_feature_rows,
)
from ml.evaluate import DATASET_SCHEMA_VERSION, TARGET_SCORE_KEY, load_historical_examples

TRAINING_LOG_PATH = Path("data/anonymized_training_rows.jsonl")
ML_GUARDRAIL_LOG_PATH = Path("data/ml_guardrail_rejections.jsonl")
ML_PERCENTAGE_MIN = 0.0
ML_PERCENTAGE_MAX = 100.0
ML_MAX_ABS_DELTA = 25.0
ML_HIGH_CONFIDENCE_ALLOW_OVERRULE_SCORE = 80.0
MIN_VALIDATED_ROWS_FOR_ML = 20
EVAL_SUMMARY_PATH = Path("data/latest_evaluation_summary.json")


def load_latest_evaluation_summary(path: Path = EVAL_SUMMARY_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def latest_ml_metric(summary: dict[str, Any] | None, metric: str) -> float | None:
    if not summary:
        return None
    metrics = summary.get("metrics")
    if not isinstance(metrics, list):
        return None
    for row in metrics:
        if isinstance(row, dict) and row.get("Model") == "ML model":
            value = row.get(metric)
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def log_ml_guardrail_rejection(
    subject: Subject,
    deterministic_pred: SubjectPrediction,
    ml_percentage: float | None,
    reason: str,
    path: Path = ML_GUARDRAIL_LOG_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "subject": subject.name,
        "reason": reason,
        "deterministic_percentage": round(deterministic_pred.predicted_final_percentage, 3),
        "deterministic_confidence_score": round(deterministic_pred.confidence_score, 3),
        "ml_percentage": None if ml_percentage is None else round(float(ml_percentage), 3),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def should_use_ml_prediction(
    subject: Subject,
    deterministic_pred: SubjectPrediction,
    ml_percentage: float | None,
) -> tuple[bool, str]:
    del subject  # reserved for future subject-level guardrails
    if ml_percentage is None:
        return False, "ml_none"
    try:
        ml_value = float(ml_percentage)
    except (TypeError, ValueError):
        return False, "ml_non_numeric"
    if math.isnan(ml_value):
        return False, "ml_nan"
    if ml_value < ML_PERCENTAGE_MIN or ml_value > ML_PERCENTAGE_MAX:
        return False, "ml_out_of_range"

    abs_delta = abs(deterministic_pred.predicted_final_percentage - ml_value)
    if (
        abs_delta > ML_MAX_ABS_DELTA
        and deterministic_pred.confidence_score < ML_HIGH_CONFIDENCE_ALLOW_OVERRULE_SCORE
    ):
        return False, f"delta_{abs_delta:.1f}_over_{ML_MAX_ABS_DELTA:.1f}"
    return True, "accepted"


def prediction_from_percentage(subject: Subject, predicted_percentage: float) -> SubjectPrediction:
    deterministic = predict(subject)
    predicted_final = max(0.0, min(100.0, predicted_percentage))
    predicted_grade = grade_from_score(predicted_final)
    next_threshold = next((low for grade, low, _ in GRADE_BOUNDS if grade == predicted_grade + 1), None)

    deterministic.predicted_final_percentage = predicted_final
    deterministic.predicted_grade = predicted_grade
    deterministic.gap_to_next_grade = 0.0 if next_threshold is None else max(0.0, next_threshold - predicted_final)
    return deterministic


def default_subject(name: str = "New Subject") -> Subject:
    default_raw_scores = [72.0, 68.0, 75.0, 80.0]
    return Subject(
        name=name,
        test_scores=default_raw_scores,
        assessment_dates=[],
        ia_progress_pct=55,
        ia_estimated_score=70,
        exam_weight=0.65,
        ia_weight=0.35,
        remaining_exam_count=2,
        remaining_exam_weights=[],
        expected_remaining_exam_avg=75,
        target_grade=6,
        test_score_obtained=default_raw_scores,
        test_score_max=[100.0] * len(default_raw_scores),
        ia_score_obtained=70.0,
        ia_score_max=100.0,
    )


def _score_rows_for_subject(subject: Subject) -> list[tuple[float, float]]:
    if subject.test_score_obtained and subject.test_score_max:
        return [
            (float(score), float(max_score))
            for score, max_score in zip(subject.test_score_obtained, subject.test_score_max)
            if max_score > 0 and 0 <= score <= max_score
        ]
    return [(float(score), 100.0) for score in subject.test_scores]


def parse_weight_list(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    weights: List[float] = []
    for p in parts:
        try:
            val = float(p)
            if val > 0:
                weights.append(val)
        except ValueError:
            continue
    return weights


def parse_dates(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    valid_dates: List[str] = []
    for p in parts:
        parsed = parse_iso_date(p)
        if parsed is not None:
            valid_dates.append(parsed.isoformat())
    return valid_dates


def scenario_subject(
    subject: Subject,
    next_test_score_obtained: float,
    next_test_score_max: float,
    ia_estimate_obtained: float,
    ia_estimate_max: float,
) -> Subject:
    next_test_pct = raw_to_pct(next_test_score_obtained, next_test_score_max)
    ia_pct = raw_to_pct(ia_estimate_obtained, ia_estimate_max)
    return Subject(
        name=subject.name,
        test_scores=subject.test_scores + [next_test_pct],
        assessment_dates=subject.assessment_dates,
        ia_progress_pct=subject.ia_progress_pct,
        ia_estimated_score=ia_pct,
        exam_weight=subject.exam_weight,
        ia_weight=subject.ia_weight,
        remaining_exam_count=subject.remaining_exam_count,
        remaining_exam_weights=subject.remaining_exam_weights,
        expected_remaining_exam_avg=subject.expected_remaining_exam_avg,
        target_grade=subject.target_grade,
        test_score_obtained=subject.test_score_obtained + [next_test_score_obtained],
        test_score_max=subject.test_score_max + [next_test_score_max],
        ia_score_obtained=ia_estimate_obtained,
        ia_score_max=ia_estimate_max,
    )


def scenario_chart_data(subject: Subject, next_test_score: float) -> dict[str, list[float | str]]:
    labels = [f"T{i + 1}" for i in range(len(subject.test_scores))]
    labels.append("Next")
    for i in range(max(0, subject.remaining_exam_count - 1)):
        labels.append(f"R{i + 1}")

    history: list[float] = subject.test_scores + [float("nan")] * (len(labels) - len(subject.test_scores))
    projection_tail = [next_test_score] + [subject.expected_remaining_exam_avg] * max(
        0, subject.remaining_exam_count - 1
    )
    projected_path: list[float] = (
        [float("nan")] * len(subject.test_scores) + projection_tail
        if projection_tail
        else [float("nan")] * len(labels)
    )
    return {"Point": labels, "Score history": history, "Projected path": projected_path}


def build_anonymized_training_row(subject: Subject, final_percentage: float) -> dict[str, float | str]:
    features = build_features(subject)
    row: dict[str, float | str] = {"schema_version": DATASET_SCHEMA_VERSION}
    row.update({key: float(features.get(key, 0.0)) for key in features})
    row[TARGET_SCORE_KEY] = max(0.0, min(100.0, float(final_percentage)))
    return row


def append_training_rows(rows: list[dict[str, float | str]], path: Path = TRAINING_LOG_PATH) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _clean_pdf_text(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def build_plan_pdf(subjects: list[Subject]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    def write_line(height: float, text: str) -> None:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_width, height, _clean_pdf_text(text))

    pdf.set_font("Helvetica", "B", 18)
    write_line(10, "IB Study Plan Summary")
    pdf.set_font("Helvetica", "", 11)
    write_line(8, "Friendly overview of where you stand and what to focus on next.")
    pdf.ln(2)

    for idx, subject in enumerate(subjects, start=1):
        pred = predict(subject)
        required_exam_avg = pred.needed_exam_avg_for_next
        target_text = (
            "You have already reached grade 7."
            if required_exam_avg is None
            else f"To reach the next grade, aim for about {required_exam_avg:.1f}% on remaining exams."
        )
        scores_text = ", ".join(f"{score:.0f}%" for score in subject.test_scores) if subject.test_scores else "No scores yet"

        pdf.set_font("Helvetica", "B", 14)
        write_line(8, f"{idx}. {subject.name}")
        pdf.set_font("Helvetica", "", 11)
        write_line(
            7,
            (
                f"Current forecast: Grade {pred.predicted_grade} ({pred.predicted_final_percentage:.1f}%). "
                f"Confidence: {pred.confidence_level}."
            ),
        )
        write_line(7, f"Recent scores: {scores_text}.")
        write_line(7, f"Target: Grade {subject.target_grade}. {target_text}")
        pdf.ln(1)

    return bytes(pdf.output(dest="S"))


def model_options(subjects: list[Subject]) -> tuple[dict[str, Any], bool]:
    load_result = load_model_bundle() if subjects else None
    eval_summary = load_latest_evaluation_summary()
    model_state = "experimental"
    if isinstance(eval_summary, dict):
        summary_state = eval_summary.get("model_state")
        if summary_state in {"ready", "experimental"}:
            model_state = summary_state

    examples = load_historical_examples(Path("data/historical_examples.csv"))
    validated_row_count = len(examples)
    if validated_row_count < MIN_VALIDATED_ROWS_FOR_ML:
        return {
            "source": "deterministic_fallback",
            "bundle": None,
            "status_badge": "insufficient_training_rows",
            "artifact_reason": None if load_result is None else load_result.reason,
            "validated_row_count": validated_row_count,
            "min_rows_required": MIN_VALIDATED_ROWS_FOR_ML,
            "model_state": "experimental",
            "evaluation_summary": eval_summary,
        }, False

    if load_result is not None and load_result.status == ArtifactLoadStatus.VALID:
        return {
            "source": "artifact",
            "bundle": load_result.bundle,
            "status_badge": "artifact_valid",
            "validated_row_count": validated_row_count,
            "min_rows_required": MIN_VALIDATED_ROWS_FOR_ML,
            "model_state": model_state,
            "evaluation_summary": eval_summary,
        }, True

    if validated_row_count >= 4:
        bundle = train_models_from_feature_rows(
            [ex.features for ex in examples],
            [ex.actual_final_score for ex in examples],
        )
        if bundle is not None:
            if load_result is not None and load_result.status == ArtifactLoadStatus.INCOMPATIBLE:
                return {
                    "source": "historical_examples",
                    "bundle": bundle,
                    "status_badge": "historical_bundle",
                    "artifact_reason": load_result.reason,
                    "validated_row_count": validated_row_count,
                    "min_rows_required": MIN_VALIDATED_ROWS_FOR_ML,
                    "model_state": model_state,
                    "evaluation_summary": eval_summary,
                }, True
            return {
                "source": "historical_examples",
                "bundle": bundle,
                "status_badge": "historical_bundle",
                "validated_row_count": validated_row_count,
                "min_rows_required": MIN_VALIDATED_ROWS_FOR_ML,
                "model_state": model_state,
                "evaluation_summary": eval_summary,
            }, True

    status_badge = "fallback_only"
    artifact_reason = None
    if load_result is not None and load_result.status == ArtifactLoadStatus.INCOMPATIBLE:
        status_badge = "artifact_incompatible"
        artifact_reason = load_result.reason

    return {
        "source": "deterministic_fallback",
        "bundle": None,
        "status_badge": status_badge,
        "artifact_reason": artifact_reason,
        "validated_row_count": validated_row_count,
        "min_rows_required": MIN_VALIDATED_ROWS_FOR_ML,
        "model_state": model_state,
        "evaluation_summary": eval_summary,
    }, False


def init_state() -> None:
    if "subjects" not in st.session_state:
        st.session_state.subjects = [subject_to_dict(default_subject("Physics HL"))]


def main() -> None:
    st.set_page_config(page_title="IB Grade Planner", page_icon="🎯")
    st.title("🎯 IB Grade Planner")
    st.caption("A simple way to see your likely grades and what to focus on next.")

    init_state()

    st.subheader("Your subjects")
    add_col, export_col = st.columns([1, 2])
    with add_col:
        if st.button("➕ Add subject"):
            st.session_state.subjects.append(
                subject_to_dict(default_subject(f"Subject {len(st.session_state.subjects) + 1}"))
            )
            st.rerun()

    with export_col:
        export_payload = build_plan_pdf([subject_from_dict(s) for s in st.session_state.subjects if isinstance(s, dict)])
        st.download_button(
            "📄 Export plan as PDF",
            data=export_payload,
            file_name="ib_study_plan.pdf",
            mime="application/pdf",
        )

    uploaded = st.file_uploader("Import saved plan (JSON)", type=["json"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            imported = json.load(uploaded)
            if isinstance(imported, list):
                parsed_subjects = [subject_to_dict(subject_from_dict(item)) for item in imported]
                if parsed_subjects:
                    st.session_state.subjects = parsed_subjects
                    st.success("Plan loaded successfully.")
            else:
                st.error("That file format is not supported. Please upload a plan JSON file.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"We couldn't open that file: {exc}")

    if not st.session_state.subjects:
        st.info("Add at least one subject to begin.")
        return

    st.markdown("---")
    updated_subjects: List[dict[str, Any]] = []

    for idx, raw_subject in enumerate(st.session_state.subjects):
        subject = subject_from_dict(raw_subject)
        with st.expander(f"{idx + 1}. {subject.name}", expanded=True):
            name = st.text_input("Subject name", value=subject.name, key=f"name_{idx}")
            st.markdown("**Historical tests/papers (raw score inputs)**")
            score_rows = _score_rows_for_subject(subject)
            add_test = st.button("➕ Add test/paper", key=f"add_test_{idx}")
            remove_last_test = st.button("➖ Remove last test/paper", key=f"remove_test_{idx}")
            if add_test:
                score_rows.append((0.0, 100.0))
            if remove_last_test and score_rows:
                score_rows = score_rows[:-1]

            score_obtained: list[float] = []
            score_max: list[float] = []
            for test_idx, (default_score, default_max) in enumerate(score_rows):
                row_cols = st.columns([2, 2, 3])
                test_max = row_cols[1].number_input(
                    f"Test {test_idx + 1} max",
                    min_value=0.1,
                    value=max(0.1, float(default_max)),
                    step=1.0,
                    key=f"test_max_{idx}_{test_idx}",
                )
                test_score = row_cols[0].number_input(
                    f"Test {test_idx + 1} score",
                    min_value=0.0,
                    max_value=float(test_max),
                    value=max(0.0, min(float(default_score), float(test_max))),
                    step=1.0,
                    key=f"test_score_{idx}_{test_idx}",
                )
                row_cols[2].caption(f"Converted: {raw_to_pct(float(test_score), float(test_max)):.1f}%")
                score_obtained.append(float(test_score))
                score_max.append(float(test_max))

            test_scores = [
                raw_to_pct(score, max_score)
                for score, max_score in zip(score_obtained, score_max)
            ]
            dates_raw = st.text_input(
                "Test dates (optional, format YYYY-MM-DD)",
                value=", ".join(subject.assessment_dates),
                key=f"dates_{idx}",
            )
            assessment_dates = parse_dates(dates_raw)

            c1, c2, c3 = st.columns(3)
            ia_progress = c1.slider(
                "IA completion (%)",
                0,
                100,
                int(subject.ia_progress_pct),
                key=f"prog_{idx}",
            )
            unknown_ia = c2.checkbox(
                "I don't know my IA score yet",
                value=subject.ia_estimated_score is None,
                key=f"ia_unknown_{idx}",
            )

            if unknown_ia:
                ia_estimated_score = None
                ia_score_obtained = None
                ia_score_max = None
                c3.number_input(
                    "Estimated IA raw score",
                    min_value=0.0,
                    value=0.0,
                    disabled=True,
                    key=f"ia_est_{idx}",
                )
            else:
                default_ia_score = (
                    70.0
                    if subject.ia_score_obtained is None
                    else float(subject.ia_score_obtained)
                )
                default_ia_max = 100.0 if subject.ia_score_max is None else float(subject.ia_score_max)
                ia_max_input = c2.number_input(
                    "IA component max",
                    min_value=0.1,
                    value=max(0.1, default_ia_max),
                    step=1.0,
                    key=f"ia_max_{idx}",
                )
                ia_score_input = c3.number_input(
                    "Estimated IA raw score",
                    min_value=0.0,
                    max_value=float(ia_max_input),
                    value=max(0.0, min(default_ia_score, float(ia_max_input))),
                    step=1.0,
                    key=f"ia_est_{idx}",
                )
                ia_score_obtained = float(ia_score_input)
                ia_score_max = float(ia_max_input)
                ia_estimated_score = raw_to_pct(ia_score_obtained, ia_score_max)
                c1.caption(f"IA estimate converts to {ia_estimated_score:.1f}%")

            w1, w2, w3 = st.columns([1, 1, 1])
            exam_input = w1.number_input(
                "Exam weight",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(subject.exam_weight),
                key=f"exam_w_{idx}",
            )
            ia_input = w2.number_input(
                "IA weight",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(subject.ia_weight),
                key=f"ia_w_{idx}",
            )
            exam_weight, ia_weight = normalize_weights(exam_input, ia_input)
            w3.write(f"Auto-balanced: **Exam {exam_weight:.2f} / IA {ia_weight:.2f}**")

            rem1, rem2 = st.columns([1, 2])
            remaining_exam_count = rem1.number_input(
                "How many exams are left?",
                min_value=0,
                max_value=12,
                value=int(subject.remaining_exam_count),
                step=1,
                key=f"remaining_count_{idx}",
            )
            remaining_weights_raw = rem2.text_input(
                "Optional: exam weight split (comma-separated)",
                value=", ".join(str(w) for w in subject.remaining_exam_weights),
                key=f"remaining_weights_{idx}",
            )
            remaining_exam_weights = parse_weight_list(remaining_weights_raw)

            t1, t2 = st.columns(2)
            target_grade = t1.number_input(
                "Your goal grade",
                min_value=1,
                max_value=7,
                value=int(subject.target_grade),
                step=1,
                key=f"target_grade_{idx}",
            )
            expected_remaining_exam_avg = t2.slider(
                "Expected average on remaining exams (%)",
                0,
                100,
                int(subject.expected_remaining_exam_avg),
                key=f"expected_exam_avg_{idx}",
            )
            final_known = st.checkbox(
                "I already know my final percentage (optional)",
                value=raw_subject.get("final_percentage") is not None,
                key=f"final_known_{idx}",
            )
            if final_known:
                final_percentage = st.number_input(
                    "Final percentage (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(raw_subject.get("final_percentage", 70.0)),
                    step=1.0,
                    key=f"final_percentage_{idx}",
                )
            else:
                final_percentage = None

            if st.button("🗑️ Remove subject", key=f"remove_{idx}"):
                continue

            if not test_scores:
                st.warning("Add at least one historical test/paper score.")

            updated_subjects.append(
                subject_to_dict(
                    Subject(
                        name=name,
                        test_scores=test_scores,
                        assessment_dates=assessment_dates,
                        ia_progress_pct=float(ia_progress),
                        ia_estimated_score=ia_estimated_score,
                        exam_weight=exam_weight,
                        ia_weight=ia_weight,
                        remaining_exam_count=int(remaining_exam_count),
                        remaining_exam_weights=remaining_exam_weights,
                        expected_remaining_exam_avg=float(expected_remaining_exam_avg),
                        target_grade=int(target_grade),
                        test_score_obtained=score_obtained,
                        test_score_max=score_max,
                        ia_score_obtained=ia_score_obtained,
                        ia_score_max=ia_score_max,
                    )
                )
            )
            if final_percentage is not None:
                updated_subjects[-1]["final_percentage"] = float(final_percentage)

    st.session_state.subjects = updated_subjects
    valid_subjects = [
        subject_from_dict(s)
        for s in st.session_state.subjects
        if isinstance(s, dict) and s.get("test_scores")
    ]

    if not valid_subjects:
        st.info("No valid subjects to predict yet.")
        return

    mode = st.radio(
        "Forecast view",
        options=["Quick estimate", "Data-learned estimate", "Side-by-side comparison"],
        horizontal=True,
        help="Quick estimate uses the rules-based model. Data-learned estimate uses ML when available.",
    )
    ml_runtime, ml_available = model_options(valid_subjects)
    st.caption(f"ML status badge: `{ml_runtime.get('status_badge', 'fallback_only')}`")
    eval_summary = ml_runtime.get("evaluation_summary")
    if isinstance(eval_summary, dict):
        eval_dataset = eval_summary.get("dataset_size", "n/a")
        eval_mae = latest_ml_metric(eval_summary, "MAE")
        eval_rmse = latest_ml_metric(eval_summary, "RMSE")
        eval_ts = eval_summary.get("timestamp_utc", "unknown")
        state = eval_summary.get("model_state", ml_runtime.get("model_state", "experimental"))
        st.caption(
            "Latest evaluation summary: "
            f"rows={eval_dataset}, MAE={eval_mae if eval_mae is not None else 'n/a'}, "
            f"RMSE={eval_rmse if eval_rmse is not None else 'n/a'}, "
            f"model_state={state}, timestamp={eval_ts}"
        )

    if ml_runtime.get("status_badge") == "insufficient_training_rows":
        st.warning(
            "ML mode disabled: need at least "
            f"{ml_runtime.get('min_rows_required', MIN_VALIDATED_ROWS_FOR_ML)} validated rows "
            f"(found {ml_runtime.get('validated_row_count', 0)})."
        )
    if ml_runtime.get("status_badge") == "artifact_incompatible":
        st.warning(
            "Saved ML artifact is incompatible with this app version. "
            "Falling back to deterministic estimates unless retraining data is available."
        )
    if ml_runtime.get("model_state") == "experimental":
        st.warning(
            "Model is currently marked experimental because calibration checks are missing or out of range. "
            "Treat ML forecasts as directional guidance only."
        )
    if ml_runtime.get("artifact_reason"):
        st.caption(f"Artifact compatibility reason: `{ml_runtime['artifact_reason']}`")
    if ml_runtime.get("status_badge") == "historical_bundle":
        st.info("Using an in-session model retrained from historical examples.")

    if mode != "Quick estimate" and not ml_available:
        st.warning("ML is not ready yet, so we are showing the rules-based estimate for now.")
    exportable_rows = []
    for raw_subject in st.session_state.subjects:
        if not isinstance(raw_subject, dict) or raw_subject.get("final_percentage") is None:
            continue
        subject = subject_from_dict(raw_subject)
        if not subject.test_scores:
            continue
        exportable_rows.append(build_anonymized_training_row(subject, float(raw_subject["final_percentage"])))

    st.caption("Optional: this logs anonymous training rows for improving the ML model.")
    export_disabled = not exportable_rows
    if st.button("🧾 Log anonymized training rows", disabled=export_disabled):
        append_training_rows(exportable_rows)
        st.success(f"Saved {len(exportable_rows)} anonymized row(s) to `{TRAINING_LOG_PATH}`.")
    if export_disabled:
        st.info("Set at least one final percentage to enable this export.")

    st.markdown("---")
    st.subheader("Forecast by subject")

    summary_rows = []
    selected_predictions: dict[str, SubjectPrediction] = {}
    for subject in valid_subjects:
        deterministic_pred = predict(subject)
        ml_percentage: float | None = None
        ml_value: float | None = None
        if ml_runtime["source"] == "artifact":
            ml_percentage = predict_with_bundle(ml_runtime["bundle"], build_features(subject))
        elif ml_runtime["source"] == "historical_examples":
            ml_percentage = predict_with_bundle(ml_runtime["bundle"], build_features(subject))
        if ml_percentage is not None:
            try:
                ml_value = float(ml_percentage)
            except (TypeError, ValueError):
                ml_value = None
        ml_allowed, ml_guardrail_reason = should_use_ml_prediction(subject, deterministic_pred, ml_percentage)
        if not ml_allowed:
            log_ml_guardrail_rejection(subject, deterministic_pred, ml_percentage, ml_guardrail_reason)
        ml_pred = prediction_from_percentage(subject, ml_percentage) if ml_allowed else deterministic_pred
        pred = deterministic_pred if mode == "Quick estimate" else ml_pred
        if mode == "Side-by-side comparison":
            pred = ml_pred if ml_allowed else deterministic_pred

        selected_predictions[subject.name] = pred
        ci_pct = f"{pred.ci_confidence_level * 100:.0f}%"
        prediction_text = (
            f"Grade {pred.predicted_grade}, {pred.predicted_final_percentage:.0f}% "
            f"(CI {ci_pct}: {pred.ci_low:.1f}%–{pred.ci_high:.1f}%) "
            f"— {pred.confidence_level} confidence"
        )
        needed_text = (
            "Already Grade 7"
            if pred.needed_exam_avg_for_next is None
            else f"{pred.needed_exam_avg_for_next:.1f}% exam avg"
        )
        if pred.confidence_level == "Low" and pred.needed_exam_avg_for_next is not None:
            needed_text += " (uncertain estimate: gather more recent test evidence)"
        summary_rows.append(
            {
                "Subject": subject.name,
                "Prediction": prediction_text,
                "Current weighted standing": round(pred.current_weighted_standing, 1),
                "Predicted final %": round(pred.predicted_final_percentage, 1),
                "Predicted IB grade": pred.predicted_grade,
                "IA confidence": pred.ia_confidence_label,
                "Overall confidence": f"{pred.confidence_level} ({pred.confidence_score:.0f}/100)",
                "Trend": f"{pred.trend_label} ({pred.trend_summary})",
                "Projection band": f"{pred.projected_range[0]:.1f}%–{pred.projected_range[1]:.1f}%",
                "Confidence interval": f"{ci_pct}: {pred.ci_low:.1f}%–{pred.ci_high:.1f}%",
                "Needed for next grade": needed_text,
            }
        )
        if mode == "Side-by-side comparison":
            summary_rows[-1]["Deterministic %"] = round(deterministic_pred.predicted_final_percentage, 1)
            summary_rows[-1]["ML %"] = round(ml_value, 1) if ml_value is not None and not math.isnan(ml_value) else "n/a"
            summary_rows[-1]["|Δ|"] = round(
                abs(deterministic_pred.predicted_final_percentage - ml_value),
                2,
            ) if ml_value is not None and not math.isnan(ml_value) else "n/a"
            summary_rows[-1]["Fallback reason"] = "none" if ml_allowed else ml_guardrail_reason
            summary_rows[-1]["ML status"] = "active" if ml_allowed else "rejected_by_guardrail"
        elif mode == "Data-learned estimate":
            summary_rows[-1]["ML status"] = "active" if ml_allowed else "rejected_by_guardrail"

    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    st.subheader("Risk check and what-if planner")
    for idx, subject in enumerate(valid_subjects):
        baseline_pred = selected_predictions.get(subject.name, predict(subject))
        with st.container(border=True):
            st.markdown(f"**{subject.name}**")
            alerts = risk_alerts(subject, baseline_pred)
            if alerts:
                for alert in alerts:
                    st.error(f"**{alert.title}**\n\n{alert.reason}\n\n{alert.threshold_text}")
            else:
                st.success("No immediate risk flags right now.")

            sc1, sc2 = st.columns(2)
            next_test_default_raw = (
                float(subject.test_score_obtained[-1])
                if subject.test_score_obtained
                else float(subject.expected_remaining_exam_avg)
            )
            next_test_default_max = (
                float(subject.test_score_max[-1])
                if subject.test_score_max
                else 100.0
            )
            next_test_max = sc1.number_input(
                "Scenario next test max",
                min_value=0.1,
                value=max(0.1, next_test_default_max),
                step=1.0,
                key=f"scenario_next_test_max_{idx}",
            )
            next_test_score = sc1.number_input(
                "Scenario next test score",
                min_value=0.0,
                max_value=float(next_test_max),
                value=max(0.0, min(next_test_default_raw, float(next_test_max))),
                step=1.0,
                key=f"scenario_next_test_score_{idx}",
            )

            scenario_ia_default_raw = (
                float(subject.ia_score_obtained) if subject.ia_score_obtained is not None else 70.0
            )
            scenario_ia_default_max = (
                float(subject.ia_score_max) if subject.ia_score_max is not None else 100.0
            )
            scenario_ia_max = sc2.number_input(
                "Scenario IA max",
                min_value=0.1,
                value=max(0.1, scenario_ia_default_max),
                step=1.0,
                key=f"scenario_ia_max_{idx}",
            )
            scenario_ia = sc2.number_input(
                "Scenario IA score",
                min_value=0.0,
                max_value=float(scenario_ia_max),
                value=max(0.0, min(scenario_ia_default_raw, float(scenario_ia_max))),
                step=1.0,
                key=f"scenario_ia_score_{idx}",
            )

            sim_subject = scenario_subject(
                subject,
                float(next_test_score),
                float(next_test_max),
                float(scenario_ia),
                float(scenario_ia_max),
            )
            sim_pred = predict(sim_subject)
            sim_ci_pct = f"{sim_pred.ci_confidence_level * 100:.0f}%"
            st.info(
                "Scenario result: "
                f"{sim_pred.predicted_final_percentage:.1f}% (Grade {sim_pred.predicted_grade}) · "
                f"{sim_ci_pct} CI {sim_pred.ci_low:.1f}%–{sim_pred.ci_high:.1f}% · "
                f"{sim_pred.confidence_level} confidence ({sim_pred.confidence_score:.0f}/100)"
            )

            target_threshold = next(
                (low for grade, low, _ in GRADE_BOUNDS if grade == sim_subject.target_grade), None
            )
            if target_threshold is not None:
                scenario_required_exam = required_remaining_exam_average(
                    sim_subject.test_scores,
                    sim_subject.remaining_exam_count,
                    sim_subject.remaining_exam_weights,
                    target_threshold,
                    0.0 if sim_subject.ia_estimated_score is None else float(sim_subject.ia_estimated_score),
                    sim_subject.exam_weight,
                    sim_subject.ia_weight,
                )
                fixed_exam_component = exam_component_from_trajectory(
                    sim_subject.test_scores,
                    sim_subject.remaining_exam_count,
                    sim_subject.remaining_exam_weights,
                    sim_subject.expected_remaining_exam_avg,
                )
                scenario_required_ia = needed_ia_quality_for_target(
                    target_threshold,
                    fixed_exam_component,
                    sim_subject.exam_weight,
                    sim_subject.ia_weight,
                )
                req_exam_text = (
                    "n/a"
                    if scenario_required_exam is None
                    else f"{scenario_required_exam:.1f}% avg on remaining exams"
                )
                req_ia_text = "n/a" if scenario_required_ia is None else f"{scenario_required_ia:.1f}% IA"
                st.caption(
                    f"Updated requirement for Grade {sim_subject.target_grade}: "
                    f"{req_exam_text}; {req_ia_text}."
                )

            chart_data = scenario_chart_data(
                subject,
                raw_to_pct(float(next_test_score), float(next_test_max)),
            )
            st.line_chart(chart_data, x="Point", y=["Score history", "Projected path"], height=190)

    st.subheader("Action plan for your target grades")
    for subject in valid_subjects:
        st.markdown(f"**{subject.name}**")
        target_threshold = next(
            (low for grade, low, _ in GRADE_BOUNDS if grade == subject.target_grade), None
        )
        ia_for_calc = subject.ia_estimated_score if subject.ia_estimated_score is not None else 0.0
        if target_threshold is None:
            st.write("Target grade is invalid.")
            continue

        required_exam_avg = required_remaining_exam_average(
            subject.test_scores,
            subject.remaining_exam_count,
            subject.remaining_exam_weights,
            target_threshold,
            ia_for_calc,
            subject.exam_weight,
            subject.ia_weight,
        )

        fixed_exam_component = exam_component_from_trajectory(
            subject.test_scores,
            subject.remaining_exam_count,
            subject.remaining_exam_weights,
            subject.expected_remaining_exam_avg,
        )
        required_ia = needed_ia_quality_for_target(
            target_threshold, fixed_exam_component, subject.exam_weight, subject.ia_weight
        )

        if required_exam_avg is None:
            st.write("Not enough exam-weight information to calculate a required exam average.")
        elif required_exam_avg > 100:
            st.error(
                f"Need {required_exam_avg:.1f}% avg on remaining exams — impossible under current assumptions."
            )
        elif required_exam_avg < 0:
            st.success("You are already on track for this target.")
        else:
            st.info(f"Need {required_exam_avg:.1f}% avg on remaining exams.")

        if required_ia is None:
            st.write("IA path is unavailable because IA weight is zero.")
        else:
            current_ia = ia_for_calc
            delta = required_ia - current_ia
            if required_ia > 100:
                st.error(
                    f"With {subject.expected_remaining_exam_avg:.0f}% exam trajectory, required IA is {required_ia:.1f}% (not feasible)."
                )
            elif required_ia < 0:
                st.success(
                    f"With {subject.expected_remaining_exam_avg:.0f}% exam trajectory, IA requirement is already secured."
                )
            elif delta >= 0:
                st.write(
                    f"Need +{delta:.1f}% IA (target IA {required_ia:.1f}%) if exam trajectory stays at {subject.expected_remaining_exam_avg:.0f}%."
                )
            else:
                st.write(
                    f"Can drop {-delta:.1f}% IA (to {required_ia:.1f}%) if exam trajectory stays at {subject.expected_remaining_exam_avg:.0f}%."
                )

    st.subheader("Simple advice")
    for subject in valid_subjects:
        pred = selected_predictions.get(subject.name, predict(subject))
        if pred.confidence_level == "Low":
            advice = (
                "This estimate has low confidence. Add more recent tests before making major study decisions."
            )
        elif pred.trend_label == "declining":
            advice = "Your recent trend is falling. Focus this week on your weakest topics."
        else:
            advice = "Your trend is stable enough. Focus on topics that can give the biggest score jump."
        st.write(f"**{subject.name}:** {advice}")

    st.subheader("Where to focus first")
    ranking = []
    for subject in valid_subjects:
        pred = selected_predictions.get(subject.name, predict(subject))
        effort_metric = pred.gap_to_next_grade
        exam_need_metric = pred.needed_exam_avg_for_next or 0.0
        ranking.append(
            {
                "Subject": subject.name,
                "Current grade": pred.predicted_grade,
                "Gap to next threshold": round(pred.gap_to_next_grade, 1),
                "Required exam avg for next": None
                if pred.needed_exam_avg_for_next is None
                else round(pred.needed_exam_avg_for_next, 1),
                "Priority score": round(effort_metric + max(0.0, exam_need_metric - 70) * 0.15, 2),
            }
        )

    ranking.sort(
        key=lambda r: (
            -r["Gap to next threshold"],
            -(r["Required exam avg for next"] if r["Required exam avg for next"] is not None else -1),
        )
    )
    st.dataframe(ranking, use_container_width=True, hide_index=True)

    with st.expander("How the forecast works"):
        st.markdown(
            """
            - Recent test scores are weighted slightly more heavily.
            - Trend analysis compares recent moving averages (3 to 5 tests) to classify momentum.
            - Predicted outcome uses IB-style weighting: exam + final IA estimate.
            - Projection bands adapt conservatively/optimistically from trend slope.
            - IA progress is used only for confidence/risk messaging.
            - Overall confidence combines evidence volume, volatility, IA progress, and assessment recency.
            - Priority ranking surfaces subjects with the largest climb to the next grade.
            """
        )


if __name__ == "__main__":
    main()
