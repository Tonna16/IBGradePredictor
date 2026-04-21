import math
from dataclasses import dataclass
from typing import List, Tuple

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
class PredictionResult:
    weighted_test_avg: float
    ia_effective_score: float
    final_composite: float
    predicted_grade: int


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
    # More recent scores matter slightly more
    n = len(scores)
    weights = list(range(1, n + 1))
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def predict(
    test_scores: List[float],
    ia_progress_pct: float,
    ia_quality_pct: float,
    test_weight: float,
    ia_weight: float,
) -> PredictionResult:
    test_avg = weighted_average(test_scores)
    ia_effective = ia_quality_pct * (ia_progress_pct / 100.0)
    final_score = test_avg * test_weight + ia_effective * ia_weight
    final_score = max(0.0, min(100.0, final_score))
    return PredictionResult(
        weighted_test_avg=test_avg,
        ia_effective_score=ia_effective,
        final_composite=final_score,
        predicted_grade=grade_from_score(final_score),
    )


def needed_test_avg_for_target(
    target_score: float,
    ia_progress_pct: float,
    ia_quality_pct: float,
    test_weight: float,
    ia_weight: float,
) -> float:
    ia_effective = ia_quality_pct * (ia_progress_pct / 100.0)
    needed = (target_score - ia_effective * ia_weight) / test_weight
    return max(0.0, min(100.0, needed))


def needed_ia_quality_for_target(
    target_score: float,
    test_avg: float,
    ia_progress_pct: float,
    test_weight: float,
    ia_weight: float,
) -> float:
    progress_factor = max(0.01, ia_progress_pct / 100.0)
    needed_effective_ia = (target_score - test_avg * test_weight) / ia_weight
    needed_quality = needed_effective_ia / progress_factor
    return max(0.0, min(100.0, needed_quality))


def parse_scores(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    scores = []
    for p in parts:
        try:
            val = float(p)
            if 0 <= val <= 100:
                scores.append(val)
        except ValueError:
            continue
    return scores


def main() -> None:
    st.set_page_config(page_title="Grade Trajectory Predictor", page_icon="📈")
    st.title("📈 Grade Trajectory Predictor")
    st.caption("Predict likely final IB grade from current test performance and IA progress.")

    with st.sidebar:
        st.header("Model settings")
        test_weight = st.slider("Test weight", 0.1, 0.9, 0.65, 0.05)
        ia_weight = 1.0 - test_weight
        st.write(f"IA weight: **{ia_weight:.2f}**")

    col1, col2 = st.columns(2)

    with col1:
        raw_scores = st.text_area(
            "Test scores (comma-separated, 0-100)",
            value="72, 68, 75, 80",
            help="Example: 72, 68, 75, 80",
        )
        ia_progress = st.slider("IA progress (%)", 0, 100, 55)

    with col2:
        ia_quality = st.slider("Current IA quality estimate (%)", 0, 100, 70)
        student_goal = st.selectbox("Target grade", [7, 6, 5, 4, 3, 2, 1], index=1)

    scores = parse_scores(raw_scores)
    if not scores:
        st.warning("Please enter at least one valid test score between 0 and 100.")
        return

    result = predict(scores, ia_progress, ia_quality, test_weight, ia_weight)

    st.subheader("Prediction")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Predicted final grade", str(result.predicted_grade))
    kpi2.metric("Composite score", f"{result.final_composite:.1f}%")
    kpi3.metric("Weighted test average", f"{result.weighted_test_avg:.1f}%")

    st.progress(min(100, int(round(result.final_composite))))

    next_threshold = next_grade_threshold(result.predicted_grade)
    st.subheader("Improvement guidance")

    if next_threshold is None:
        st.success("You're already projected at Grade 7. Focus on consistency.")
    else:
        need_points = next_threshold - result.final_composite
        needed_test = needed_test_avg_for_target(
            next_threshold, ia_progress, ia_quality, test_weight, ia_weight
        )
        needed_ia = needed_ia_quality_for_target(
            next_threshold, result.weighted_test_avg, ia_progress, test_weight, ia_weight
        )

        st.info(
            f"To reach Grade {result.predicted_grade + 1}, you need about "
            f"**{need_points:.1f}** more composite points."
        )
        st.write(
            f"- Required test average (if IA estimate stays same): **{needed_test:.1f}%**\n"
            f"- Required IA quality estimate (if tests stay same): **{needed_ia:.1f}%**"
        )

    target_threshold = next(low for grade, low, _ in GRADE_BOUNDS if grade == student_goal)
    needed_test_goal = needed_test_avg_for_target(
        target_threshold, ia_progress, ia_quality, test_weight, ia_weight
    )
    st.subheader(f"Your Grade {student_goal} plan")
    if result.predicted_grade >= student_goal:
        st.success(f"You're currently on track for Grade {student_goal} or higher.")
    else:
        st.write(
            f"To target Grade {student_goal}, aim for a test average near "
            f"**{needed_test_goal:.1f}%** with current IA assumptions."
        )

    with st.expander("How this forecast works"):
        st.markdown(
            """
            - Recent test scores are slightly weighted more heavily.
            - IA contributes based on both quality estimate and completion progress.
            - Composite score is mapped to grade boundaries.
            """
        )


if __name__ == "__main__":
    main()
