from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _binary_f1(y_true: Sequence[bool], y_pred: Sequence[bool]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


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


def _set_f1(gold: Set[str], pred: Set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold.intersection(pred))
    precision = _safe_div(tp, len(pred))
    recall = _safe_div(tp, len(gold))
    return _safe_div(2 * precision * recall, precision + recall)


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "binary_accuracy": 0.0,
            "binary_f1": 0.0,
            "bdi_mae": 0.0,
            "symptom_f1_at_4": 0.0,
            "item_f1_macro_at_1": 0.0,
            "item_mae": 0.0,
            "headline_f1": 0.0,
            "avg_turns_to_decision": 0.0,
            "risk_recall": None,
            "class_counts": {
                "depressed_true": 0,
                "control_true": 0,
                "depressed_pred": 0,
                "control_pred": 0,
            },
            "binary_f1_defined": False,
            "risk_recall_defined": False,
        }

    y_true = [bool(row["y_true"]) for row in records]
    y_pred = [bool(row["y_pred"]) for row in records]
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

    risk_records = [row for row in records if bool(row["risk_true"])]
    risk_recall_defined = bool(risk_records)
    if risk_records:
        risk_recall = sum(1 for row in risk_records if bool(row["risk_pred"])) / len(risk_records)
    else:
        risk_recall = None

    binary_accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(records)
    binary_f1 = _binary_f1(y_true, y_pred)
    depressed_true = sum(1 for value in y_true if value)
    control_true = len(y_true) - depressed_true
    depressed_pred = sum(1 for value in y_pred if value)
    control_pred = len(y_pred) - depressed_pred
    binary_f1_defined = depressed_true > 0 and control_true > 0
    if binary_f1 > 0.0 and item_f1_macro > 0.0:
        headline_f1 = 2.0 * binary_f1 * item_f1_macro / (binary_f1 + item_f1_macro)
    else:
        headline_f1 = 0.0

    payload: Dict[str, Any] = {
        "binary_accuracy": round(binary_accuracy, 4),
        "binary_f1": round(binary_f1, 4),
        "bdi_mae": round(bdi_mae, 4),
        # Backward compatible key; now aliases full-item macro F1@1.
        "symptom_f1_at_4": round(item_f1_macro, 4),
        "item_f1_macro_at_1": round(item_f1_macro, 4),
        "item_mae": round(item_mae, 4),
        "headline_f1": round(headline_f1, 4),
        "avg_turns_to_decision": round(avg_turns, 4),
        "risk_recall": (round(float(risk_recall), 4) if risk_recall is not None else None),
        "class_counts": {
            "depressed_true": int(depressed_true),
            "control_true": int(control_true),
            "depressed_pred": int(depressed_pred),
            "control_pred": int(control_pred),
        },
        "binary_f1_defined": bool(binary_f1_defined),
        "risk_recall_defined": bool(risk_recall_defined),
    }
    return payload
