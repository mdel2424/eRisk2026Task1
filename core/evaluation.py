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


def _set_f1(gold: Set[str], pred: Set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold.intersection(pred))
    precision = _safe_div(tp, len(pred))
    recall = _safe_div(tp, len(gold))
    return _safe_div(2 * precision * recall, precision + recall)


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {
            "binary_accuracy": 0.0,
            "binary_f1": 0.0,
            "bdi_mae": 0.0,
            "symptom_f1_at_4": 0.0,
            "avg_turns_to_decision": 0.0,
            "risk_recall": 0.0,
        }

    y_true = [bool(row["y_true"]) for row in records]
    y_pred = [bool(row["y_pred"]) for row in records]
    bdi_mae = sum(abs(float(row["bdi_true"]) - float(row["bdi_pred"])) for row in records) / len(records)
    symptom_f1 = sum(_set_f1(set(row["symptoms_true"]), set(row["symptoms_pred"])) for row in records) / len(records)
    avg_turns = sum(float(row["turns"]) for row in records) / len(records)

    risk_records = [row for row in records if bool(row["risk_true"])]
    if risk_records:
        risk_recall = sum(1 for row in risk_records if bool(row["risk_pred"])) / len(risk_records)
    else:
        risk_recall = 1.0

    binary_accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(records)
    binary_f1 = _binary_f1(y_true, y_pred)

    return {
        "binary_accuracy": round(binary_accuracy, 4),
        "binary_f1": round(binary_f1, 4),
        "bdi_mae": round(bdi_mae, 4),
        "symptom_f1_at_4": round(symptom_f1, 4),
        "avg_turns_to_decision": round(avg_turns, 4),
        "risk_recall": round(risk_recall, 4),
    }
