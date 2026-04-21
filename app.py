import json
from dataclasses import asdict, dataclass
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
    ia_progress_pct: float
    ia_estimated_score: float | None
    exam_weight: float
    ia_weight: float


@dataclass
class SubjectPrediction:
    current_weighted_standing: float
    predicted_final_percentage: float
    predicted_grade: int
    gap_to_next_grade: float
    needed_exam_avg_for_next: float | None


def default_subject(name: str = "New Subject") -> Subject:
    return Subject(
        name=name,
        test_scores=[72, 68, 75, 80],
        ia_progress_pct=55,
        ia_estimated_score=70,
        exam_weight=0.65,
        ia_weight=0.35,
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


def needed_exam_avg_for_target(
    target_score: float,
    ia_estimated_score: float,
    exam_weight: float,
    ia_weight: float,
) -> float:
    if exam_weight <= 0:
        return 100.0
    needed = (target_score - ia_estimated_score * ia_weight) / exam_weight
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


def normalize_weights(exam_weight: float, ia_weight: float) -> tuple[float, float]:
    total = exam_weight + ia_weight
    if total <= 0:
        return 0.65, 0.35
    return exam_weight / total, ia_weight / total


def predict_subject(subject: Subject) -> SubjectPrediction:
    exam_avg = weighted_average(subject.test_scores)
    ia_est = subject.ia_estimated_score or 0.0

    current_ia_contrib = ia_est * (subject.ia_progress_pct / 100)
    current = exam_avg * subject.exam_weight + current_ia_contrib * subject.ia_weight
    current = max(0.0, min(100.0, current))

    predicted_final = exam_avg * subject.exam_weight + ia_est * subject.ia_weight
    predicted_final = max(0.0, min(100.0, predicted_final))
    grade = grade_from_score(predicted_final)

    next_threshold = next_grade_threshold(grade)
    if next_threshold is None:
        gap_to_next = 0.0
        needed_exam = None
    else:
        gap_to_next = max(0.0, next_threshold - predicted_final)
        needed_exam = needed_exam_avg_for_target(
            next_threshold, ia_est, subject.exam_weight, subject.ia_weight
        )

    return SubjectPrediction(
        current_weighted_standing=current,
        predicted_final_percentage=predicted_final,
        predicted_grade=grade,
        gap_to_next_grade=gap_to_next,
        needed_exam_avg_for_next=needed_exam,
    )


def subject_to_dict(subject: Subject) -> dict[str, Any]:
    return asdict(subject)


def subject_from_dict(payload: dict[str, Any]) -> Subject:
    exam_w = float(payload.get("exam_weight", 0.65))
    ia_w = float(payload.get("ia_weight", 0.35))
    exam_w, ia_w = normalize_weights(exam_w, ia_w)
    return Subject(
        name=str(payload.get("name", "Imported Subject")),
        test_scores=[float(s) for s in payload.get("test_scores", []) if 0 <= float(s) <= 100],
        ia_progress_pct=float(payload.get("ia_progress_pct", 0)),
        ia_estimated_score=(
            None
            if payload.get("ia_estimated_score") is None
            else float(payload.get("ia_estimated_score"))
        ),
        exam_weight=exam_w,
        ia_weight=ia_w,
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

            c1, c2, c3 = st.columns(3)
            ia_progress = c1.slider("IA progress %", 0, 100, int(subject.ia_progress_pct), key=f"prog_{idx}")
            unknown_ia = c2.checkbox(
                "IA estimate unknown",
                value=subject.ia_estimated_score is None,
                key=f"ia_unknown_{idx}",
            )

            if unknown_ia:
                ia_estimated_score = None
                c3.number_input(
                    "IA estimate %",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    disabled=True,
                    key=f"ia_est_{idx}",
                )
            else:
                default_ia = 70.0 if subject.ia_estimated_score is None else float(subject.ia_estimated_score)
                ia_estimated_score = c3.number_input(
                    "IA estimate %",
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

            if st.button("🗑️ Remove subject", key=f"remove_{idx}"):
                continue

            if not test_scores:
                st.warning("Add at least one valid test score (0-100).")

            updated_subjects.append(
                subject_to_dict(
                    Subject(
                        name=name,
                        test_scores=test_scores,
                        ia_progress_pct=float(ia_progress),
                        ia_estimated_score=ia_estimated_score,
                        exam_weight=exam_weight,
                        ia_weight=ia_weight,
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
        pred = predict_subject(subject)
        needed_text = (
            "Already Grade 7"
            if pred.needed_exam_avg_for_next is None
            else f"{pred.needed_exam_avg_for_next:.1f}% exam avg"
        )
        summary_rows.append(
            {
                "Subject": subject.name,
                "Current weighted standing": round(pred.current_weighted_standing, 1),
                "Predicted final %": round(pred.predicted_final_percentage, 1),
                "Predicted IB grade": pred.predicted_grade,
                "Needed for next grade": needed_text,
            }
        )

    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    st.subheader("Priority ranking")
    ranking = []
    for subject in valid_subjects:
        pred = predict_subject(subject)
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
            - Current standing includes only completed IA progress.
            - Predicted final assumes IA reaches the estimated final score.
            - Priority ranking surfaces subjects with the largest climb to the next grade.
            """
        )


if __name__ == "__main__":
    main()
