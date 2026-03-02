from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _f1_from_binary_arrays(y_true: Sequence[bool], y_pred: Sequence[bool]) -> tuple[float, bool]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and (not p))
    defined = (tp + fp + fn) > 0
    if not defined:
        return 0.0, False
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall), True


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "binary_accuracy": None,
            "binary_f1": None,
            "bdi_mae": 0.0,
            "symptom_f1_at_4": 0.0,
            "item_f1_macro_at_1": 0.0,
            "item_mae": 0.0,
            "headline_f1": 0.0,
            "avg_turns_to_decision": 0.0,
            "risk_recall": None,
            "binary_f1_defined": False,
            "risk_recall_defined": False,
            "metric_mode": "item_only",
        }

    bdi_mae = sum(abs(float(row["bdi_true"]) - float(row["bdi_pred"])) for row in records) / len(records)
    avg_turns = sum(float(row["turns"]) for row in records) / len(records)

    # Full 21-item evaluation (positive if score >=1) for symptom fidelity.
    per_item_f1: List[float] = []
    item_abs_errors: List[float] = []
    for item_id in range(1, 22):
        item_true_binary: List[bool] = []
        item_pred_binary: List[bool] = []
        for row in records:
            scores_true = dict(row.get("item_scores_true", {}))
            scores_pred = dict(row.get("item_scores_pred", {}))
            true_score = int(scores_true.get(str(item_id), scores_true.get(item_id, 0)) or 0)
            pred_score = int(scores_pred.get(str(item_id), scores_pred.get(item_id, 0)) or 0)
            item_true_binary.append(true_score >= 1)
            item_pred_binary.append(pred_score >= 1)
            item_abs_errors.append(abs(float(true_score) - float(pred_score)))
        item_f1, defined = _f1_from_binary_arrays(item_true_binary, item_pred_binary)
        if defined:
            per_item_f1.append(item_f1)

    item_f1_macro = sum(per_item_f1) / len(per_item_f1) if per_item_f1 else 0.0
    item_mae = sum(item_abs_errors) / len(item_abs_errors) if item_abs_errors else 0.0

    headline_f1 = item_f1_macro

    payload: Dict[str, Any] = {
        "binary_accuracy": None,
        "binary_f1": None,
        "bdi_mae": round(bdi_mae, 4),
        # Backward compatible key; now aliases full-item macro F1@1.
        "symptom_f1_at_4": round(item_f1_macro, 4),
        "item_f1_macro_at_1": round(item_f1_macro, 4),
        "item_mae": round(item_mae, 4),
        "headline_f1": round(headline_f1, 4),
        "avg_turns_to_decision": round(avg_turns, 4),
        "risk_recall": None,
        "binary_f1_defined": False,
        "risk_recall_defined": False,
        "metric_mode": "item_only",
    }
    return payload
