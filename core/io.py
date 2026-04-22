from __future__ import annotations

from dataclasses import asdict
from typing import Any

from core.predictor import Subject, normalize_weights, parse_iso_date


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
            if parse_iso_date(str(d)) is not None
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
