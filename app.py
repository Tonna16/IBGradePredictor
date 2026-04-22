import json
from typing import Any, List

import streamlit as st

from core.io import subject_from_dict, subject_to_dict
from core.predictor import (
    GRADE_BOUNDS,
    Subject,
    exam_component_from_trajectory,
    needed_ia_quality_for_target,
    normalize_weights,
    parse_iso_date,
    predict,
    required_remaining_exam_average,
    risk_alerts,
)


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


def parse_dates(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    valid_dates: List[str] = []
    for p in parts:
        parsed = parse_iso_date(p)
        if parsed is not None:
            valid_dates.append(parsed.isoformat())
    return valid_dates


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
    valid_subjects = [
        subject_from_dict(s)
        for s in st.session_state.subjects
        if isinstance(s, dict) and s.get("test_scores")
    ]

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
