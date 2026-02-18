from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

from core.state import ItemBelief

ITEM_FEATURES = [f"item_{i}" for i in range(1, 22)]
EXTRA_FEATURES = ["evidence_conf_mean", "evidence_conf_max", "evidence_count_norm", "risk_flag"]
FEATURE_NAMES = ITEM_FEATURES + EXTRA_FEATURES


@dataclass
class Contribution:
    feature: str
    value: float
    weight: float
    impact: float


@dataclass
class PredictionResult:
    predicted_bdi_score: int
    predicted_label: str
    global_confidence: float
    positive_contributions: List[Contribution]
    negative_contributions: List[Contribution]
    raw_label_score: float
    mode: str


@dataclass
class CalibratorBundle:
    mode: str
    feature_names: List[str]
    bdi_weights: List[float]
    bdi_intercept: float
    label_weights: List[float]
    label_intercept: float


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _vectorize(features: Dict[str, float], feature_names: Sequence[str]) -> List[float]:
    return [float(features.get(name, 0.0)) for name in feature_names]


def _default_bundle() -> CalibratorBundle:
    bdi_weights = [1.8 if name.startswith("item_") else 0.0 for name in FEATURE_NAMES]
    label_weights = [0.25 if name.startswith("item_") else 0.0 for name in FEATURE_NAMES]
    for idx, name in enumerate(FEATURE_NAMES):
        if name == "risk_flag":
            label_weights[idx] = 2.2
        if name == "evidence_conf_mean":
            label_weights[idx] = 0.6
        if name == "evidence_count_norm":
            label_weights[idx] = 0.4
    return CalibratorBundle(
        mode="deterministic",
        feature_names=list(FEATURE_NAMES),
        bdi_weights=bdi_weights,
        bdi_intercept=0.0,
        label_weights=label_weights,
        label_intercept=-2.2,
    )


def build_feature_vector(
    item_beliefs: Dict[int, ItemBelief],
    evidence_confidences: List[float],
    risk_flag: bool,
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for item_id in range(1, 22):
        belief = item_beliefs[item_id]
        features[f"item_{item_id}"] = float(belief.mean_score)

    if evidence_confidences:
        conf_mean = sum(evidence_confidences) / len(evidence_confidences)
        conf_max = max(evidence_confidences)
    else:
        conf_mean = 0.0
        conf_max = 0.0

    features["evidence_conf_mean"] = float(conf_mean)
    features["evidence_conf_max"] = float(conf_max)
    features["evidence_count_norm"] = min(1.0, len(evidence_confidences) / 5.0)
    features["risk_flag"] = 1.0 if risk_flag else 0.0
    return features


def fit_calibrator(records: List[Dict]) -> CalibratorBundle:
    if not records:
        return _default_bundle()

    features = [record.get("features", {}) for record in records]
    y_bdi = [float(record.get("bdi_true", 0.0)) for record in records]
    y_label = [1 if bool(record.get("y_true", False)) else 0 for record in records]
    feature_names = list(FEATURE_NAMES)
    x = [_vectorize(feature_map, feature_names) for feature_map in features]

    try:
        from sklearn.linear_model import LogisticRegression, Ridge
    except Exception:
        return _default_bundle()

    # Need minimum 10 samples for reasonably reliable model fitting.
    if len(records) < 10:
        return _default_bundle()
    # Need at least 2 distinct labels (e.g., control and depressed) for binary classification calibration.
    if len(set(y_label)) < 2:
        return _default_bundle()

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(x, y_bdi)

    logreg = LogisticRegression(max_iter=500, random_state=42)
    logreg.fit(x, y_label)

    return CalibratorBundle(
        mode="sklearn",
        feature_names=feature_names,
        bdi_weights=[float(w) for w in ridge.coef_],
        bdi_intercept=float(ridge.intercept_),
        label_weights=[float(w) for w in logreg.coef_[0]],
        label_intercept=float(logreg.intercept_[0]),
    )


def predict_with_explanations(features: Dict[str, float], bundle: CalibratorBundle) -> PredictionResult:
    x = _vectorize(features, bundle.feature_names)

    bdi_raw = bundle.bdi_intercept + sum(weight * value for weight, value in zip(bundle.bdi_weights, x))
    predicted_bdi_score = max(0, min(63, int(round(bdi_raw))))

    raw_label_score = bundle.label_intercept + sum(
        weight * value for weight, value in zip(bundle.label_weights, x)
    )
    prob_depressed = _sigmoid(raw_label_score)
    predicted_label = "depressed" if prob_depressed >= 0.5 else "control"
    global_confidence = prob_depressed if predicted_label == "depressed" else (1.0 - prob_depressed)

    contributions = [
        Contribution(
            feature=name,
            value=value,
            weight=weight,
            impact=weight * value,
        )
        for name, value, weight in zip(bundle.feature_names, x, bundle.label_weights)
    ]
    ranked = sorted(contributions, key=lambda c: c.impact, reverse=True)
    positive = [c for c in ranked if c.impact > 0][:5]
    negative = [c for c in sorted(contributions, key=lambda c: c.impact) if c.impact < 0][:5]

    return PredictionResult(
        predicted_bdi_score=predicted_bdi_score,
        predicted_label=predicted_label,
        global_confidence=max(0.0, min(1.0, global_confidence)),
        positive_contributions=positive,
        negative_contributions=negative,
        raw_label_score=raw_label_score,
        mode=bundle.mode,
    )


def save_calibrator_bundle(path: str | Path, bundle: CalibratorBundle) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(bundle), indent=2), encoding="utf-8")


def load_calibrator_bundle(path: str | Path) -> CalibratorBundle:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibratorBundle(
        mode=str(data.get("mode", "deterministic")),
        feature_names=[str(v) for v in data.get("feature_names", FEATURE_NAMES)],
        bdi_weights=[float(v) for v in data.get("bdi_weights", [])],
        bdi_intercept=float(data.get("bdi_intercept", 0.0)),
        label_weights=[float(v) for v in data.get("label_weights", [])],
        label_intercept=float(data.get("label_intercept", 0.0)),
    )


@lru_cache(maxsize=1)
def get_calibrator_bundle() -> CalibratorBundle:
    path = os.getenv("CALIBRATOR_PATH", "").strip()
    if path and Path(path).exists():
        try:
            return load_calibrator_bundle(path)
        except Exception:
            return _default_bundle()
    return _default_bundle()
