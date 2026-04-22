from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from core.predictor import LEGACY_EXAM_SIGNAL_CONFIG, Subject, predict
from ml.forecaster import FEATURE_KEYS, predict_with_bundle, train_models_from_feature_rows

DATASET_SCHEMA_VERSION = "1.0"
TARGET_SCORE_KEY = "final_percentage"
RANGED_FEATURE_KEYS = {"mean", "recent_mean", "ia_estimate", "ia_progress"}


@dataclass
class HistoricalExample:
    features: dict[str, float]
    actual_final_score: float


def _to_float(payload: object, *, default: float = 0.0) -> float:
    try:
        return float(payload)
    except (TypeError, ValueError):
        return default


def _clamp_score(score: float) -> float:
    return max(0.0, min(100.0, score))


def _coerce_required_number(row: dict[str, object], key: str) -> float | None:
    if key not in row:
        return None
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_and_parse_row(row: object) -> HistoricalExample | None:
    if not isinstance(row, dict):
        return None

    schema_version = str(row.get("schema_version", DATASET_SCHEMA_VERSION))
    if schema_version != DATASET_SCHEMA_VERSION:
        return None

    features: dict[str, float] = {}
    for key in FEATURE_KEYS:
        value = _coerce_required_number(row, key)
        if value is None:
            return None
        if key in RANGED_FEATURE_KEYS:
            value = _clamp_score(value)
        elif key in {"std", "recency_mean_days", "recency_std_days", "latest_days_ago"}:
            value = max(0.0, value)
        elif key == "test_count":
            if value < 1:
                return None
        features[key] = value

    target_value = _coerce_required_number(row, TARGET_SCORE_KEY)
    if target_value is None:
        return None
    return HistoricalExample(features=features, actual_final_score=_clamp_score(target_value))


def load_historical_examples(path: Path) -> list[HistoricalExample]:
    if not path.exists():
        return []

    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text())
    else:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    examples: list[HistoricalExample] = []
    for row in rows:
        parsed = _validate_and_parse_row(row)
        if parsed is None:
            continue
        examples.append(parsed)
    return examples


def _synthesize_scores(features: dict[str, float]) -> list[float]:
    count = max(1, int(round(_to_float(features.get("test_count", 1.0), default=1.0))))
    mean_score = _to_float(features.get("mean", 0.0))
    slope = _to_float(features.get("slope", 0.0))
    midpoint = (count - 1) / 2.0
    scores = [mean_score + slope * (i - midpoint) for i in range(count)]
    return [_clamp_score(score) for score in scores]


def _subject_from_features(features: dict[str, float], idx: int) -> Subject:
    return Subject(
        name=f"Historical #{idx}",
        test_scores=_synthesize_scores(features),
        assessment_dates=[],
        ia_progress_pct=max(0.0, min(100.0, _to_float(features.get("ia_progress", 0.0)))),
        ia_estimated_score=_clamp_score(_to_float(features.get("ia_estimate", 0.0))),
        exam_weight=0.65,
        ia_weight=0.35,
        remaining_exam_count=1,
        remaining_exam_weights=[],
        expected_remaining_exam_avg=_clamp_score(_to_float(features.get("recent_mean", 0.0))),
        target_grade=6,
    )


def _mae(actuals: list[float], preds: list[float]) -> float:
    if not actuals:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actuals, preds)) / len(actuals)


def _rmse(actuals: list[float], preds: list[float]) -> float:
    if not actuals:
        return 0.0
    return sqrt(sum((a - p) ** 2 for a, p in zip(actuals, preds)) / len(actuals))


def evaluate_models(
    examples: list[HistoricalExample],
    *,
    test_size: float = 0.3,
    seed: int = 42,
    include_rmse: bool = True,
) -> list[dict[str, float | str]]:
    if len(examples) < 4:
        return []

    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    test_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * test_size))))
    holdout = shuffled[:test_count]
    train = shuffled[test_count:]
    if len(train) < 2:
        return []

    train_x = [ex.features for ex in train]
    train_y = [ex.actual_final_score for ex in train]
    bundle = train_models_from_feature_rows(train_x, train_y)
    if bundle is None:
        return []

    holdout_actuals = [ex.actual_final_score for ex in holdout]
    legacy_preds: list[float] = []
    upgraded_preds: list[float] = []
    ml_preds: list[float] = []

    for idx, ex in enumerate(holdout):
        subject = _subject_from_features(ex.features, idx)
        legacy_pred = predict(subject, signal_config=LEGACY_EXAM_SIGNAL_CONFIG).predicted_final_percentage
        upgraded_pred = predict(subject).predicted_final_percentage
        ml_pred = predict_with_bundle(bundle, ex.features)

        legacy_preds.append(_clamp_score(legacy_pred))
        upgraded_preds.append(_clamp_score(upgraded_pred))
        ml_preds.append(_clamp_score(ml_pred))

    comparison = [
        ("Legacy deterministic model", legacy_preds),
        ("Upgraded statistical model", upgraded_preds),
        ("ML model", ml_preds),
    ]

    rows: list[dict[str, float | str]] = []
    for label, preds in comparison:
        row: dict[str, float | str] = {
            "Model": label,
            "MAE": round(_mae(holdout_actuals, preds), 3),
        }
        if include_rmse:
            row["RMSE"] = round(_rmse(holdout_actuals, preds), 3)
        rows.append(row)
    return rows


def _print_table(rows: list[dict[str, float | str]]) -> None:
    if not rows:
        print("No evaluation rows generated. Need at least 4 historical examples.")
        return

    headers = list(rows[0].keys())
    widths = {header: max(len(header), *(len(str(r[header])) for r in rows)) for header in headers}
    header_row = " | ".join(f"{h:<{widths[h]}}" for h in headers)
    divider = "-+-".join("-" * widths[h] for h in headers)
    print(header_row)
    print(divider)
    for row in rows:
        print(" | ".join(f"{str(row[h]):<{widths[h]}}" for h in headers))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MAE/RMSE on held-out historical examples.")
    parser.add_argument("--data", default="data/historical_examples.csv", help="CSV or JSON dataset path")
    parser.add_argument("--test-size", type=float, default=0.3, help="Hold-out proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    parser.add_argument("--no-rmse", action="store_true", help="Only report MAE")
    args = parser.parse_args()

    examples = load_historical_examples(Path(args.data))
    rows = evaluate_models(
        examples,
        test_size=args.test_size,
        seed=args.seed,
        include_rmse=not args.no_rmse,
    )
    _print_table(rows)


if __name__ == "__main__":
    main()
