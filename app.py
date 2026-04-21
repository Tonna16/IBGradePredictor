import json
from datetime import date, datetime
from dataclasses import asdict, dataclass
from statistics import pstdev
from typing import Any, List

import streamlit as st


GRADE_BOUNDS = [
    (7, 85, 100),
    (6, 75, 84.9999),
    (5, 65, 74.9999),
    (4, 55, 64.9999),
    (3, 45, 54.9999),
    (2, 35, 44.9999),
    (1, 0, 34.9999),
]


@dataclass
class Subject:
    name: str
    test_scores: List[float]
    assessment_dates: List[str]
    ia_progress_pct: float
    ia_estimated_score: float | None
    exam_weight: float
    ia_weight: float
    remaining_exam_count: int
    remaining_exam_weights: List[float]
    expected_remaining_exam_avg: float
    target_grade: int


@dataclass
class SubjectPrediction:
    current_weighted_standing: float
    predicted_final_percentage: float
    predicted_grade: int
    gap_to_next_grade: float
    needed_exam_avg_for_next: float | None
    needed_ia_for_next: float | None
    ia_confidence_label: str
    trend_label: str
    trend_summary: str
    projected_range: tuple[float, float]
    confidence_score: float
    confidence_level: str
    confidence_summary: str


@dataclass
class TrendInsight:
    label: str
    numeric_change: float
    summary: str
    slope_per_test: float
    tests_used: int


@dataclass
class RiskAlert:
    title: str
    reason: str
    threshold_text: str


def default_subject(name: str = "New Subject") -> Subject:
    return Subject(
        name=name,
        test_scores=[72, 68, 75, 80],
        assessment_dates=[],
        ia_progress_pct=55,
        ia_estimated_score=70,
        exam_weight=0.65,
        ia_weight=0.35,
        remaining_exam_count=2,
        remaining_exam_weights=[],
        expected_remaining_exam_avg=75,
        target_grade=6,
    )


def grade_from_score(score: float) -> int:
    for grade, low, high in GRADE_BOUNDS:
        if low <= score <= high:
            return grade
    return 1


def next_grade_threshold(current_grade: int) -> float | None:
    if current_grade >= 7:
        return None
    for grade, low, _ in GRADE_BOUNDS:
        if grade == current_grade + 1:
            return low
    return None


def weighted_average(scores: List[float]) -> float:
    if not scores:
        return 0.0
    n = len(scores)
    weights = list(range(1, n + 1))
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def needed_test_avg_for_target(
    target_score: float,
    ia_estimated_score: float,
    exam_weight: float,
    ia_weight: float,
) -> float | None:
    if exam_weight <= 0:
        return None
    needed = (target_score - ia_estimated_score * ia_weight) / exam_weight
    return max(0.0, min(100.0, needed))


def needed_ia_quality_for_target(
    target_score: float,
    exam_avg: float,
    exam_weight: float,
    ia_weight: float,
) -> float | None:
    if ia_weight <= 0:
        return None
    needed = (target_score - exam_avg * exam_weight) / ia_weight
    return max(0.0, min(100.0, needed))


def parse_scores(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    scores: List[float] = []
    for p in parts:
        try:
            val = float(p)
            if 0 <= val <= 100:
                scores.append(val)
        except ValueError:
            continue
    return scores


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


def normalize_weights(exam_weight: float, ia_weight: float) -> tuple[float, float]:
    total = exam_weight + ia_weight
    if total <= 0:
        return 0.65, 0.35
    return exam_weight / total, ia_weight / total


def ia_confidence_from_progress(ia_progress_pct: float) -> str:
    if ia_progress_pct < 35:
        return "Low confidence (IA estimate likely to move)"
    if ia_progress_pct < 70:
        return "Medium confidence"
    return "High confidence"


def moving_average(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _parse_iso_date(raw: str) -> date | None:
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_dates(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    valid_dates: List[str] = []
    for p in parts:
        parsed = _parse_iso_date(p)
        if parsed is not None:
            valid_dates.append(parsed.isoformat())
    return valid_dates


def recency_confidence(assessment_dates: List[str]) -> float:
    if not assessment_dates:
        return 0.5

    valid = [_parse_iso_date(d) for d in assessment_dates]
    valid = [d for d in valid if d is not None]
    if not valid:
        return 0.5

    latest = max(valid)
    days_old = max(0, (date.today() - latest).days)
    if days_old <= 14:
        return 1.0
    if days_old <= 45:
        return 0.75
    if days_old <= 90:
        return 0.45
    return 0.2


def confidence_from_evidence(
    test_scores: List[float], ia_progress_pct: float, assessment_dates: List[str]
) -> tuple[float, str, str]:
    test_count = len(test_scores)
    count_component = min(test_count, 6) / 6 * 35

    volatility = pstdev(test_scores) if len(test_scores) >= 2 else 18.0
    volatility_component = max(0.0, min(1.0, 1.0 - volatility / 20.0)) * 25

    progress_component = max(0.0, min(100.0, ia_progress_pct)) / 100 * 25
    recency_component = recency_confidence(assessment_dates) * 15

    score = round(count_component + volatility_component + progress_component + recency_component, 1)
    if score >= 70:
        level = "High"
    elif score >= 45:
        level = "Medium"
    else:
        level = "Low"

    summary = (
        f"{level} confidence (evidence={test_count} tests, volatility={volatility:.1f}, "
        f"IA progress={ia_progress_pct:.0f}%)"
    )
    return score, level, summary


def calculate_trend(scores: List[float], window: int = 3) -> TrendInsight | None:
    if len(scores) < 3:
        return None

    tests_used = max(3, min(window, len(scores), 5))
    recent = scores[-tests_used:]
    pivot = max(1, tests_used // 2)
    early = recent[:pivot]
    late = recent[pivot:]
    if not early or not late:
        return None

    change = moving_average(late) - moving_average(early)
    slope = (recent[-1] - recent[0]) / max(1, tests_used - 1)

    if change > 1.5:
        label = "improving"
    elif change < -1.5:
        label = "declining"
    else:
        label = "stagnant"

    summary = f"{change:+.1f}% over last {tests_used} tests"
    return TrendInsight(
        label=label,
        numeric_change=change,
        summary=summary,
        slope_per_test=slope,
        tests_used=tests_used,
    )


def projection_band(exam_avg: float, slope_per_test: float, scenario: str) -> tuple[float, float]:
    adjustments = {"conservative": -0.5, "neutral": 0.0, "optimistic": 0.5}
    bias = adjustments.get(scenario, 0.0)
    center = exam_avg + slope_per_test + bias
    spread = max(1.5, abs(slope_per_test) * 1.25 + 1.0)
    low = max(0.0, min(100.0, center - spread))
    high = max(0.0, min(100.0, center + spread))
    return low, high


def predict(subject: Subject) -> SubjectPrediction:
    exam_avg = weighted_average(subject.test_scores)
    ia_est = subject.ia_estimated_score or 0.0
    trend = calculate_trend(subject.test_scores)
    trend_label = trend.label if trend is not None else "No trend"
    trend_summary = (
        trend.summary if trend is not None else "Need at least 3 tests to classify trend."
    )
    scenario = "neutral" if trend is None else ("optimistic" if trend.slope_per_test > 0.3 else "conservative" if trend.slope_per_test < -0.3 else "neutral")
    projected_exam_low, projected_exam_high = projection_band(
        exam_avg,
        0.0 if trend is None else trend.slope_per_test,
        scenario,
    )
    projected_low = projected_exam_low * subject.exam_weight + ia_est * subject.ia_weight
    projected_high = projected_exam_high * subject.exam_weight + ia_est * subject.ia_weight
    confidence_score, confidence_level, confidence_summary = confidence_from_evidence(
        subject.test_scores, subject.ia_progress_pct, subject.assessment_dates
    )

    # IB-style weighted composition: IA contribution is not reduced by progress.
    current = exam_avg * subject.exam_weight + ia_est * subject.ia_weight
    current = max(0.0, min(100.0, current))

    predicted_final = exam_avg * subject.exam_weight + ia_est * subject.ia_weight
    predicted_final = max(0.0, min(100.0, predicted_final))
    grade = grade_from_score(predicted_final)

    next_threshold = next_grade_threshold(grade)
    if next_threshold is None:
        gap_to_next = 0.0
        needed_exam, needed_ia = None, None
    else:
        gap_to_next = max(0.0, next_threshold - predicted_final)
        needed_exam = needed_test_avg_for_target(
            next_threshold, ia_est, subject.exam_weight, subject.ia_weight
        )
        needed_ia = needed_ia_quality_for_target(
            next_threshold, exam_avg, subject.exam_weight, subject.ia_weight
        )

    return SubjectPrediction(
        current_weighted_standing=current,
        predicted_final_percentage=predicted_final,
        predicted_grade=grade,
        gap_to_next_grade=gap_to_next,
        needed_exam_avg_for_next=needed_exam,
        needed_ia_for_next=needed_ia,
        ia_confidence_label=ia_confidence_from_progress(subject.ia_progress_pct),
        trend_label=trend_label,
        trend_summary=trend_summary,
        projected_range=(projected_low, projected_high),
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        confidence_summary=confidence_summary,
    )


def max_grade_with_ia_ceiling(ia_estimated_score: float, exam_weight: float, ia_weight: float) -> tuple[int, float]:
    max_final_pct = max(0.0, min(100.0, 100.0 * exam_weight + ia_estimated_score * ia_weight))
    return grade_from_score(max_final_pct), max_final_pct


def risk_alerts(subject: Subject, prediction: SubjectPrediction) -> List[RiskAlert]:
    alerts: List[RiskAlert] = []
    trend = calculate_trend(subject.test_scores)
    volatility = pstdev(subject.test_scores) if len(subject.test_scores) >= 2 else 0.0

    if subject.ia_estimated_score is not None:
        max_grade, max_final = max_grade_with_ia_ceiling(
            subject.ia_estimated_score, subject.exam_weight, subject.ia_weight
        )
        if max_grade < subject.target_grade:
            alerts.append(
                RiskAlert(
                    title="IA ceiling limits maximum achievable grade",
                    reason=(
                        f"Even with perfect exams, ceiling is Grade {max_grade} "
                        f"({max_final:.1f}%), below target Grade {subject.target_grade}."
                    ),
                    threshold_text=(
                        f"Threshold check: max_grade={max_grade} < target_grade={subject.target_grade}"
                    ),
                )
            )

    exam_heavy_threshold = 0.75
    decline_threshold = -0.8
    slope = 0.0 if trend is None else trend.slope_per_test
    if subject.exam_weight >= exam_heavy_threshold and slope <= decline_threshold:
        alerts.append(
            RiskAlert(
                title="Over-reliance on exams while trend is weak",
                reason=(
                    f"Exam weighting is high ({subject.exam_weight:.2f}) and recent slope is "
                    f"{slope:+.2f} pts/test."
                ),
                threshold_text=(
                    f"Threshold check: exam_weight ≥ {exam_heavy_threshold:.2f} and "
                    f"slope ≤ {decline_threshold:+.2f}"
                ),
            )
        )

    volatility_threshold = 10.0
    if volatility >= volatility_threshold:
        alerts.append(
            RiskAlert(
                title="High volatility / inconsistency warning",
                reason=f"Score volatility is {volatility:.1f} across {len(subject.test_scores)} tests.",
                threshold_text=f"Threshold check: stdev={volatility:.1f} ≥ {volatility_threshold:.1f}",
            )
        )

    if prediction.confidence_level == "Low":
        alerts.append(
            RiskAlert(
                title="Forecast confidence is low",
                reason=prediction.confidence_summary,
                threshold_text="Threshold check: confidence score < 45",
            )
        )

    return alerts


def scenario_subject(subject: Subject, next_test_score: float, ia_estimate: float) -> Subject:
    return Subject(
        name=subject.name,
        test_scores=subject.test_scores + [next_test_score],
        assessment_dates=subject.assessment_dates,
        ia_progress_pct=subject.ia_progress_pct,
        ia_estimated_score=ia_estimate,
        exam_weight=subject.exam_weight,
        ia_weight=subject.ia_weight,
        remaining_exam_count=subject.remaining_exam_count,
        remaining_exam_weights=subject.remaining_exam_weights,
        expected_remaining_exam_avg=subject.expected_remaining_exam_avg,
        target_grade=subject.target_grade,
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


def needed_exam_avg_for_target(
    target_score: float,
    ia_estimated_score: float,
    exam_weight: float,
    ia_weight: float,
) -> float | None:
    # Backward-compatible alias.
    return needed_test_avg_for_target(target_score, ia_estimated_score, exam_weight, ia_weight)


def predict_subject(subject: Subject) -> SubjectPrediction:
    # Backward-compatible alias.
    return predict(subject)


def required_remaining_exam_average(
    completed_scores: List[float],
    remaining_count: int,
    remaining_weights: List[float],
    target_score: float,
    ia_score: float,
    exam_weight: float,
    ia_weight: float,
) -> float | None:
    if exam_weight <= 0:
        return None

    done_weights = [1.0] * len(completed_scores)
    rem_weights = remaining_weights[:remaining_count]
    if len(rem_weights) < remaining_count:
        rem_weights.extend([1.0] * (remaining_count - len(rem_weights)))

    sum_done = sum(s * w for s, w in zip(completed_scores, done_weights))
    weight_done = sum(done_weights)
    weight_rem = sum(rem_weights)
    if weight_rem <= 0:
        return None

    required_exam_component = (target_score - ia_score * ia_weight) / exam_weight
    needed_total_exam_points = required_exam_component * (weight_done + weight_rem)
    return (needed_total_exam_points - sum_done) / weight_rem


def exam_component_from_trajectory(
    completed_scores: List[float],
    remaining_count: int,
    remaining_weights: List[float],
    expected_remaining_avg: float,
) -> float:
    done_weights = [1.0] * len(completed_scores)
    rem_weights = remaining_weights[:remaining_count]
    if len(rem_weights) < remaining_count:
        rem_weights.extend([1.0] * (remaining_count - len(rem_weights)))
    sum_done = sum(s * w for s, w in zip(completed_scores, done_weights))
    weight_done = sum(done_weights)
    weight_rem = sum(rem_weights)
    total_weight = weight_done + weight_rem
    if total_weight <= 0:
        return 0.0
    return (sum_done + expected_remaining_avg * weight_rem) / total_weight


def subject_to_dict(subject: Subject) -> dict[str, Any]:
    return asdict(subject)


def subject_from_dict(payload: dict[str, Any]) -> Subject:
    exam_w = float(payload.get("exam_weight", 0.65))
    ia_w = float(payload.get("ia_weight", 0.35))
    exam_w, ia_w = normalize_weights(exam_w, ia_w)
    return Subject(
        name=str(payload.get("name", "Imported Subject")),
        test_scores=[float(s) for s in payload.get("test_scores", []) if 0 <= float(s) <= 100],
        assessment_dates=[
            str(d)
            for d in payload.get("assessment_dates", [])
            if _parse_iso_date(str(d)) is not None
        ],
        ia_progress_pct=float(payload.get("ia_progress_pct", 0)),
        ia_estimated_score=(
            None
            if payload.get("ia_estimated_score") is None
            else float(payload.get("ia_estimated_score"))
        ),
        exam_weight=exam_w,
        ia_weight=ia_w,
        remaining_exam_count=int(payload.get("remaining_exam_count", 2)),
        remaining_exam_weights=[
            float(w) for w in payload.get("remaining_exam_weights", []) if float(w) > 0
        ],
        expected_remaining_exam_avg=float(payload.get("expected_remaining_exam_avg", 75)),
        target_grade=max(1, min(7, int(payload.get("target_grade", 6)))),
    )


def init_state() -> None:
    if "subjects" not in st.session_state:
        st.session_state.subjects = [subject_to_dict(default_subject("Physics HL"))]


def main() -> None:
    st.set_page_config(page_title="IB Multi-Subject Predictor", page_icon="📈")
    st.title("📈 IB Multi-Subject Grade Predictor")
    st.caption("Track and forecast grades across all subjects, then prioritize where to focus.")

    init_state()

    st.subheader("Subject planner")
    add_col, export_col = st.columns([1, 2])
    with add_col:
        if st.button("➕ Add subject"):
            st.session_state.subjects.append(
                subject_to_dict(default_subject(f"Subject {len(st.session_state.subjects) + 1}"))
            )
            st.rerun()

    with export_col:
        export_payload = json.dumps(st.session_state.subjects, indent=2)
        st.download_button(
            "💾 Export plan JSON",
            data=export_payload,
            file_name="ib_subject_plan.json",
            mime="application/json",
        )

    uploaded = st.file_uploader("Import saved plan", type=["json"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            imported = json.load(uploaded)
            if isinstance(imported, list):
                parsed_subjects = [subject_to_dict(subject_from_dict(item)) for item in imported]
                if parsed_subjects:
                    st.session_state.subjects = parsed_subjects
                    st.success("Plan imported successfully.")
            else:
                st.error("Invalid JSON format. Expected a list of subjects.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to import JSON: {exc}")

    if not st.session_state.subjects:
        st.info("Add at least one subject to begin.")
        return

    st.markdown("---")
    updated_subjects: List[dict[str, Any]] = []

    for idx, raw_subject in enumerate(st.session_state.subjects):
        subject = subject_from_dict(raw_subject)
        with st.expander(f"{idx + 1}. {subject.name}", expanded=True):
            name = st.text_input("Subject name", value=subject.name, key=f"name_{idx}")
            scores_raw = st.text_input(
                "Test scores (comma-separated %)",
                value=", ".join(str(int(s)) if s.is_integer() else f"{s:.1f}" for s in subject.test_scores),
                key=f"scores_{idx}",
            )
            test_scores = parse_scores(scores_raw)
            dates_raw = st.text_input(
                "Assessment dates (optional, comma-separated YYYY-MM-DD)",
                value=", ".join(subject.assessment_dates),
                key=f"dates_{idx}",
            )
            assessment_dates = parse_dates(dates_raw)

            c1, c2, c3 = st.columns(3)
            ia_progress = c1.slider(
                "IA progress (%) — confidence only",
                0,
                100,
                int(subject.ia_progress_pct),
                key=f"prog_{idx}",
            )
            unknown_ia = c2.checkbox(
                "IA estimate unknown",
                value=subject.ia_estimated_score is None,
                key=f"ia_unknown_{idx}",
            )

            if unknown_ia:
                ia_estimated_score = None
                c3.number_input(
                    "Estimated final IA score (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    disabled=True,
                    key=f"ia_est_{idx}",
                )
            else:
                default_ia = 70.0 if subject.ia_estimated_score is None else float(subject.ia_estimated_score)
                ia_estimated_score = c3.number_input(
                    "Estimated final IA score (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_ia,
                    step=1.0,
                    key=f"ia_est_{idx}",
                )

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
            w3.write(f"Normalized: **Exam {exam_weight:.2f} / IA {ia_weight:.2f}**")

            rem1, rem2 = st.columns([1, 2])
            remaining_exam_count = rem1.number_input(
                "Exams left",
                min_value=0,
                max_value=12,
                value=int(subject.remaining_exam_count),
                step=1,
                key=f"remaining_count_{idx}",
            )
            remaining_weights_raw = rem2.text_input(
                "Expected weight per remaining exam (optional, comma-separated)",
                value=", ".join(str(w) for w in subject.remaining_exam_weights),
                key=f"remaining_weights_{idx}",
            )
            remaining_exam_weights = parse_weight_list(remaining_weights_raw)

            t1, t2 = st.columns(2)
            target_grade = t1.number_input(
                "Desired IB grade threshold",
                min_value=1,
                max_value=7,
                value=int(subject.target_grade),
                step=1,
                key=f"target_grade_{idx}",
            )
            expected_remaining_exam_avg = t2.slider(
                "Expected avg on remaining exams (%)",
                0,
                100,
                int(subject.expected_remaining_exam_avg),
                key=f"expected_exam_avg_{idx}",
            )

            if st.button("🗑️ Remove subject", key=f"remove_{idx}"):
                continue

            if not test_scores:
                st.warning("Add at least one valid test score (0-100).")

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
                    )
                )
            )

    st.session_state.subjects = updated_subjects
    valid_subjects = [subject_from_dict(s) for s in st.session_state.subjects if s.test_scores]

    if not valid_subjects:
        st.info("No valid subjects to predict yet.")
        return

    st.markdown("---")
    st.subheader("Per-subject forecast")

    summary_rows = []
    for subject in valid_subjects:
        pred = predict(subject)
        prediction_text = (
            f"Grade {pred.predicted_grade}, {pred.predicted_final_percentage:.0f}% "
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
                "Needed for next grade": needed_text,
            }
        )

    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    st.subheader("Risk & scenario intervention planner")
    for idx, subject in enumerate(valid_subjects):
        baseline_pred = predict(subject)
        with st.container(border=True):
            st.markdown(f"**{subject.name}**")
            alerts = risk_alerts(subject, baseline_pred)
            if alerts:
                for alert in alerts:
                    st.error(f"**{alert.title}**\n\n{alert.reason}\n\n{alert.threshold_text}")
            else:
                st.success("No immediate risk flags under current thresholds.")

            sc1, sc2 = st.columns(2)
            next_test_default = int(round(subject.expected_remaining_exam_avg))
            next_test_score = sc1.slider(
                "What if next test is X%?",
                0,
                100,
                max(0, min(100, next_test_default)),
                key=f"scenario_next_test_{idx}",
            )
            scenario_ia_default = (
                int(round(subject.ia_estimated_score))
                if subject.ia_estimated_score is not None
                else 70
            )
            scenario_ia = sc2.slider(
                "What if IA estimate becomes Y%?",
                0,
                100,
                max(0, min(100, scenario_ia_default)),
                key=f"scenario_ia_{idx}",
            )

            sim_subject = scenario_subject(subject, float(next_test_score), float(scenario_ia))
            sim_pred = predict(sim_subject)
            st.info(
                "Scenario result: "
                f"{sim_pred.predicted_final_percentage:.1f}% (Grade {sim_pred.predicted_grade}) · "
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
                    float(scenario_ia),
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
                    f"Updated target requirements for Grade {sim_subject.target_grade}: "
                    f"{req_exam_text}; {req_ia_text}."
                )

            chart_data = scenario_chart_data(subject, float(next_test_score))
            st.line_chart(chart_data, x="Point", y=["Score history", "Projected path"], height=190)

    st.subheader("Grade-target action plans")
    for subject in valid_subjects:
        st.markdown(f"**{subject.name}**")
        target_threshold = next(
            (low for grade, low, _ in GRADE_BOUNDS if grade == subject.target_grade), None
        )
        ia_for_calc = subject.ia_estimated_score if subject.ia_estimated_score is not None else 0.0
        if target_threshold is None:
            st.write("Invalid target grade.")
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
            st.write("Not enough exam-weight information to compute a required exam average.")
        elif required_exam_avg > 100:
            st.error(
                f"Need {required_exam_avg:.1f}% avg on remaining exams — impossible under current assumptions."
            )
        elif required_exam_avg < 0:
            st.success("Target already secured from current trajectory (required avg is below 0%).")
        else:
            st.info(f"Need {required_exam_avg:.1f}% avg on remaining exams.")

        if required_ia is None:
            st.write("IA path unavailable because IA weight is zero.")
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

    st.subheader("Advice")
    for subject in valid_subjects:
        pred = predict(subject)
        if pred.confidence_level == "Low":
            advice = (
                "Low confidence: collect more test evidence (and recent assessments) before "
                "making high-stakes study decisions."
            )
        elif pred.trend_label == "declining":
            advice = "Performance trend is declining; prioritize revision on weak units this week."
        else:
            advice = "Confidence is usable; focus effort on topics with the highest mark gain potential."
        st.write(f"**{subject.name}:** {advice}")

    st.subheader("Priority ranking")
    ranking = []
    for subject in valid_subjects:
        pred = predict(subject)
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

    with st.expander("How this forecast works"):
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
