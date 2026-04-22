from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import date
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle

from core.predictor import Subject, parse_iso_date, predict

MODEL_ARTIFACT_PATH = Path("artifacts/forecaster.pkl")
MODEL_SCHEMA_VERSION = "1"
FEATURE_KEYS = [
    "mean",
    "recent_mean",
    "std",
    "slope",
    "test_count",
    "ia_estimate",
    "ia_progress",
    "recency_mean_days",
    "recency_std_days",
    "latest_days_ago",
]


@dataclass
class ModelBundle:
    linear_model: LinearRegression
    tree_model: RandomForestRegressor


class ArtifactLoadStatus(str, Enum):
    VALID = "artifact_valid"
    MISSING = "artifact_missing"
    INCOMPATIBLE = "artifact_incompatible"
    MALFORMED = "artifact_malformed"


@dataclass
class BundleLoadResult:
    bundle: ModelBundle | None
    status: ArtifactLoadStatus
    reason: str | None = None


def _score_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _score_std(values: list[float]) -> float:
    return pstdev(values) if len(values) >= 2 else 0.0


def _least_squares_slope(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    n = len(scores)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(scores) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, scores))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _recency_day_offsets(assessment_dates: list[str]) -> list[int]:
    valid_dates = [parse_iso_date(d) for d in assessment_dates]
    valid_dates = [d for d in valid_dates if d is not None]
    today = date.today()
    return [max(0, (today - d).days) for d in valid_dates]


def build_features(subject: Subject) -> dict[str, float]:
    scores = subject.test_scores
    recent_window = scores[-3:] if len(scores) >= 3 else scores
    recency_offsets = _recency_day_offsets(subject.assessment_dates)

    return {
        "mean": _score_mean(scores),
        "recent_mean": _score_mean(recent_window),
        "std": _score_std(scores),
        "slope": _least_squares_slope(scores),
        "test_count": float(len(scores)),
        "ia_estimate": 0.0 if subject.ia_estimated_score is None else float(subject.ia_estimated_score),
        "ia_progress": float(subject.ia_progress_pct),
        "recency_mean_days": _score_mean([float(x) for x in recency_offsets]),
        "recency_std_days": _score_std([float(x) for x in recency_offsets]),
        "latest_days_ago": float(min(recency_offsets)) if recency_offsets else 999.0,
    }


def _feature_matrix(feature_rows: Iterable[dict[str, float]]) -> list[list[float]]:
    return [[float(row.get(k, 0.0)) for k in FEATURE_KEYS] for row in feature_rows]


def train_models_from_feature_rows(
    feature_rows: list[dict[str, float]],
    targets: list[float],
) -> ModelBundle | None:
    if len(feature_rows) < 2 or len(feature_rows) != len(targets):
        return None

    x = _feature_matrix(feature_rows)
    y_vals = [float(y) for y in targets]

    linear_model = LinearRegression()
    linear_model.fit(x, y_vals)

    tree_model = RandomForestRegressor(n_estimators=200, random_state=42)
    tree_model.fit(x, y_vals)
    return ModelBundle(linear_model=linear_model, tree_model=tree_model)


def predict_with_bundle(bundle: ModelBundle, features: dict[str, float]) -> float:
    row = _feature_matrix([features])
    linear_pred = float(bundle.linear_model.predict(row)[0])
    tree_pred = float(bundle.tree_model.predict(row)[0])
    combined = (linear_pred + tree_pred) / 2.0
    return max(0.0, min(100.0, combined))


def train_models(subjects: list[Subject], artifact_path: Path = MODEL_ARTIFACT_PATH) -> ModelBundle | None:
    if not subjects:
        return None

    x_rows = []
    y_vals = []
    for subject in subjects:
        if not subject.test_scores:
            continue
        x_rows.append(build_features(subject))
        y_vals.append(predict(subject).predicted_final_percentage)

    bundle = train_models_from_feature_rows(x_rows, y_vals)
    if bundle is None:
        return None

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as f:
        pickle.dump(
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_keys": FEATURE_KEYS,
                "linear_model": bundle.linear_model,
                "tree_model": bundle.tree_model,
            },
            f,
        )

    return bundle


def load_model_bundle(artifact_path: Path = MODEL_ARTIFACT_PATH) -> BundleLoadResult:
    if not artifact_path.exists():
        return BundleLoadResult(bundle=None, status=ArtifactLoadStatus.MISSING, reason="artifact_missing")

    with artifact_path.open("rb") as f:
        try:
            payload = pickle.load(f)
        except Exception:  # noqa: BLE001
            return BundleLoadResult(bundle=None, status=ArtifactLoadStatus.MALFORMED, reason="artifact_unreadable")

    schema_version = payload.get("schema_version")
    if schema_version != MODEL_SCHEMA_VERSION:
        return BundleLoadResult(
            bundle=None,
            status=ArtifactLoadStatus.INCOMPATIBLE,
            reason=f"schema_version_mismatch:{schema_version!r}!={MODEL_SCHEMA_VERSION!r}",
        )

    feature_keys = payload.get("feature_keys")
    if feature_keys != FEATURE_KEYS:
        return BundleLoadResult(
            bundle=None,
            status=ArtifactLoadStatus.INCOMPATIBLE,
            reason="feature_keys_mismatch",
        )

    linear_model = payload.get("linear_model")
    tree_model = payload.get("tree_model")
    if linear_model is None or tree_model is None:
        return BundleLoadResult(bundle=None, status=ArtifactLoadStatus.MALFORMED, reason="missing_model_fields")

    return BundleLoadResult(
        bundle=ModelBundle(linear_model=linear_model, tree_model=tree_model),
        status=ArtifactLoadStatus.VALID,
        reason=None,
    )


def predict_with_model(features: dict[str, float], artifact_path: Path = MODEL_ARTIFACT_PATH) -> float | None:
    load_result = load_model_bundle(artifact_path)
    if load_result.bundle is None:
        return None

    return predict_with_bundle(load_result.bundle, features)
