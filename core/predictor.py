from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import exp
from statistics import NormalDist, pstdev
from typing import List

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
    ci_low: float
    ci_high: float
    ci_confidence_level: float


@dataclass(frozen=True)
class ExamSignalConfig:
    name: str
    weighting_method: str
    trend_method: str
    alpha: float = 0.35


@dataclass
class TrendInsight:
    label: str
    numeric_change: float
    summary: str
    slope_per_test: float
    tests_used: int
    intercept: float | None = None
    projection_center: float | None = None
    method: str = "legacy"


@dataclass
class RiskAlert:
    title: str
    reason: str
    threshold_text: str


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


LEGACY_EXAM_SIGNAL_CONFIG = ExamSignalConfig(
    name="legacy",
    weighting_method="linear_ramp",
    trend_method="window_delta",
    alpha=0.35,
)
UPGRADED_EXAM_SIGNAL_CONFIG = ExamSignalConfig(
    name="upgraded_math",
    weighting_method="exponential_decay",
    trend_method="least_squares",
    alpha=0.35,
)
EXAM_SIGNAL_CONFIGS = {
    LEGACY_EXAM_SIGNAL_CONFIG.name: LEGACY_EXAM_SIGNAL_CONFIG,
    UPGRADED_EXAM_SIGNAL_CONFIG.name: UPGRADED_EXAM_SIGNAL_CONFIG,
}


def weighted_average(scores: List[float], *, config: ExamSignalConfig = UPGRADED_EXAM_SIGNAL_CONFIG) -> float:
    if not scores:
        return 0.0

    n = len(scores)
    if config.weighting_method == "linear_ramp":
        weights = list(range(1, n + 1))
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    alpha = max(0.0, config.alpha)
    weights = [exp(alpha * i) for i in range(n)]
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


def weighted_average_legacy(scores: List[float]) -> float:
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


def least_squares_trend(scores: List[float]) -> tuple[float, float] | None:
    if len(scores) < 2:
        return None

    n = len(scores)
    x_vals = list(range(n))
    x_mean = sum(x_vals) / n
    y_mean = sum(scores) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, scores))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)
    if denominator == 0:
        return 0.0, y_mean

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def parse_iso_date(raw: str) -> date | None:
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def recency_confidence(assessment_dates: List[str]) -> float:
    if not assessment_dates:
        return 0.5

    valid = [parse_iso_date(d) for d in assessment_dates]
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
        method="legacy",
    )


def calculate_regression_trend(scores: List[float]) -> TrendInsight | None:
    if len(scores) < 3:
        return None

    trend = least_squares_trend(scores)
    if trend is None:
        return None

    slope, intercept = trend
    first_fit = intercept
    last_fit = intercept + slope * (len(scores) - 1)
    change = last_fit - first_fit
    projection_center = intercept + slope * len(scores)

    if change > 1.5:
        label = "improving"
    elif change < -1.5:
        label = "declining"
    else:
        label = "stagnant"

    summary = (
        f"regression slope {slope:+.2f} pts/test, intercept {intercept:.1f}, "
        f"fit change {change:+.1f}% across {len(scores)} tests"
    )
    return TrendInsight(
        label=label,
        numeric_change=change,
        summary=summary,
        slope_per_test=slope,
        tests_used=len(scores),
        intercept=intercept,
        projection_center=projection_center,
        method="least_squares",
    )


def projection_band(
    exam_avg: float,
    slope_per_test: float,
    scenario: str,
    *,
    projection_center: float | None = None,
) -> tuple[float, float]:
    adjustments = {"conservative": -0.5, "neutral": 0.0, "optimistic": 0.5}
    bias = adjustments.get(scenario, 0.0)
    center = (exam_avg + slope_per_test) if projection_center is None else projection_center
    center += bias
    spread = max(1.5, abs(slope_per_test) * 1.25 + 1.0)
    low = max(0.0, min(100.0, center - spread))
    high = max(0.0, min(100.0, center + spread))
    return low, high


def z_based_confidence_interval(
    scores: List[float],
    predicted_final_percentage: float,
    exam_weight: float,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    clamped_confidence = min(0.999, max(0.50, confidence_level))
    z_score = NormalDist().inv_cdf((1.0 + clamped_confidence) / 2.0)

    sample_size = max(1, len(scores))
    observed_std = pstdev(scores) if len(scores) >= 2 else 12.0
    std_error = observed_std / (sample_size**0.5)
    final_percentage_std_error = std_error * max(0.0, exam_weight)

    margin = z_score * final_percentage_std_error
    ci_low = max(0.0, min(100.0, predicted_final_percentage - margin))
    ci_high = max(0.0, min(100.0, predicted_final_percentage + margin))
    return ci_low, ci_high, clamped_confidence


def predict(subject: Subject, signal_config: ExamSignalConfig = UPGRADED_EXAM_SIGNAL_CONFIG) -> SubjectPrediction:
    use_legacy_fallback = len(subject.test_scores) < 3
    active_config = LEGACY_EXAM_SIGNAL_CONFIG if use_legacy_fallback else signal_config

    exam_avg = weighted_average(subject.test_scores, config=active_config)
    ia_est = subject.ia_estimated_score or 0.0
    if active_config.trend_method == "least_squares":
        trend = calculate_regression_trend(subject.test_scores)
    else:
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
        projection_center=None if trend is None else trend.projection_center,
    )
    projected_low = projected_exam_low * subject.exam_weight + ia_est * subject.ia_weight
    projected_high = projected_exam_high * subject.exam_weight + ia_est * subject.ia_weight
    confidence_score, confidence_level, confidence_summary = confidence_from_evidence(
        subject.test_scores, subject.ia_progress_pct, subject.assessment_dates
    )

    current = exam_avg * subject.exam_weight + ia_est * subject.ia_weight
    current = max(0.0, min(100.0, current))

    predicted_final = exam_avg * subject.exam_weight + ia_est * subject.ia_weight
    predicted_final = max(0.0, min(100.0, predicted_final))
    ci_low, ci_high, ci_confidence_level = z_based_confidence_interval(
        subject.test_scores,
        predicted_final,
        subject.exam_weight,
    )
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
        ci_low=ci_low,
        ci_high=ci_high,
        ci_confidence_level=ci_confidence_level,
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

    next_threshold = next_grade_threshold(prediction.predicted_grade)
    if (
        next_threshold is not None
        and prediction.ci_low <= next_threshold <= prediction.ci_high
    ):
        alerts.append(
            RiskAlert(
                title="Next-grade threshold sits inside forecast interval",
                reason=(
                    f"The {prediction.ci_confidence_level:.0%} interval is "
                    f"{prediction.ci_low:.1f}%–{prediction.ci_high:.1f}%, and the next grade "
                    f"threshold ({next_threshold:.1f}%) lies inside that range."
                ),
                threshold_text=(
                    "Threshold check: ci_low ≤ next_grade_threshold ≤ ci_high"
                ),
            )
        )

    return alerts


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


def needed_exam_avg_for_target(
    target_score: float,
    ia_estimated_score: float,
    exam_weight: float,
    ia_weight: float,
) -> float | None:
    return needed_test_avg_for_target(target_score, ia_estimated_score, exam_weight, ia_weight)


def predict_subject(subject: Subject) -> SubjectPrediction:
    return predict(subject)


def predict_side_by_side(subject: Subject) -> dict[str, SubjectPrediction]:
    return {
        LEGACY_EXAM_SIGNAL_CONFIG.name: predict(subject, LEGACY_EXAM_SIGNAL_CONFIG),
        UPGRADED_EXAM_SIGNAL_CONFIG.name: predict(subject, UPGRADED_EXAM_SIGNAL_CONFIG),
    }
