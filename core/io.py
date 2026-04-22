from __future__ import annotations

from dataclasses import asdict
from typing import Any

from core.predictor import Subject, normalize_weights, parse_iso_date, raw_to_pct


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_raw_pairs(payload: dict[str, Any]) -> tuple[list[float], list[float]]:
    obtained_raw = payload.get("test_score_obtained", [])
    max_raw = payload.get("test_score_max", [])
    if isinstance(obtained_raw, list) and isinstance(max_raw, list) and obtained_raw and max_raw:
        obtained = [_coerce_float(v) for v in obtained_raw]
        max_scores = [_coerce_float(v) for v in max_raw]
        if None not in obtained and None not in max_scores:
            valid_pairs: list[tuple[float, float]] = []
            for score, max_score in zip(obtained, max_scores):
                score_f = float(score)
                max_f = float(max_score)
                try:
                    raw_to_pct(score_f, max_f)
                except ValueError:
                    continue
                valid_pairs.append((score_f, max_f))
            if valid_pairs:
                return [score for score, _ in valid_pairs], [max_score for _, max_score in valid_pairs]
    legacy_scores = [_coerce_float(s) for s in payload.get("test_scores", [])]
    valid_legacy = [float(s) for s in legacy_scores if s is not None and 0 <= s <= 100]
    return valid_legacy, [100.0] * len(valid_legacy)


def _read_ia_raw_pair(payload: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    raw_score = _coerce_float(payload.get("ia_score_obtained"))
    raw_max = _coerce_float(payload.get("ia_score_max"))
    if raw_score is not None and raw_max is not None:
        try:
            pct = raw_to_pct(raw_score, raw_max)
        except ValueError:
            pct = None
        if pct is not None:
            return raw_score, raw_max, pct

    legacy_pct = payload.get("ia_estimated_score")
    if legacy_pct is None:
        return None, None, None
    pct = _coerce_float(legacy_pct)
    if pct is None:
        return None, None, None
    pct = max(0.0, min(100.0, pct))
    return pct, 100.0, pct


def subject_to_dict(subject: Subject) -> dict[str, Any]:
    return asdict(subject)


def subject_from_dict(payload: dict[str, Any]) -> Subject:
    exam_w = float(payload.get("exam_weight", 0.65))
    ia_w = float(payload.get("ia_weight", 0.35))
    exam_w, ia_w = normalize_weights(exam_w, ia_w)

    test_score_obtained, test_score_max = _read_raw_pairs(payload)
    test_scores = [raw_to_pct(score, max_score) for score, max_score in zip(test_score_obtained, test_score_max)]
    ia_score_obtained, ia_score_max, ia_estimated_score = _read_ia_raw_pair(payload)

    return Subject(
        name=str(payload.get("name", "Imported Subject")),
        test_scores=test_scores,
        assessment_dates=[
            str(d)
            for d in payload.get("assessment_dates", [])
            if parse_iso_date(str(d)) is not None
        ],
        ia_progress_pct=float(payload.get("ia_progress_pct", 0)),
        ia_estimated_score=ia_estimated_score,
        exam_weight=exam_w,
        ia_weight=ia_w,
        remaining_exam_count=int(payload.get("remaining_exam_count", 2)),
        remaining_exam_weights=[
            float(w) for w in payload.get("remaining_exam_weights", []) if float(w) > 0
        ],
        expected_remaining_exam_avg=float(payload.get("expected_remaining_exam_avg", 75)),
        target_grade=max(1, min(7, int(payload.get("target_grade", 6)))),
        test_score_obtained=test_score_obtained,
        test_score_max=test_score_max,
        ia_score_obtained=ia_score_obtained,
        ia_score_max=ia_score_max,
    )
