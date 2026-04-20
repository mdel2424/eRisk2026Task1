from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from core.bdi_modules import MODULE_NAMES, MODULE_TO_ITEMS, NODE_ALLOWED_MODULES, choose_target_module, modules_for_item
from core.state import AgentState, ConversationThreadState, NextAction, RouteDecision


STYLE_CYCLE = ("gentle_probe", "clarify_frequency", "functional_impact")
FOLLOWUP_POLICIES = {
    "evidence_followup_same_item",
    "evidence_followup_same_module",
    "conversation_contrastive_pivot",
}
FOLLOWUP_WINDOW = 3
RECENT_COUNT_WINDOW = 6
THREAD_MAX_QUESTIONS = 3
THREAD_EXIT_DENIALS = 2

GLOBAL_IG_WEIGHT = 1.0
GLOBAL_ENTROPY_WEIGHT = 0.60
GLOBAL_UNCOVERED_BONUS = 0.30
GLOBAL_MODULE_UNRESOLVED_WEIGHT = 0.20
ITEM_REPETITION_PENALTY = 0.75
MODULE_REPETITION_PENALTY = 0.35
SAME_ITEM_EXPECTED_MIN = 1.0
SAME_ITEM_EXPECTED_MAX = 2.35
RISK_GLOBAL_DAMPENER = 8.0
RISK_REENTRY_LOOKBACK = 6
RISK_REENTRY_TOTAL_BDI_THRESHOLD = 18.0
RISK_REENTRY_CORE_SUPPORT_THRESHOLD = 3
RISK_REENTRY_BONUS = 0.85
MODULE_SATURATION_MIN_SUPPORTED = 2
MODULE_SATURATION_MAX_UNRESOLVED_RATIO = 0.40

MANDATORY_COVERAGE_ITEMS = {14, 16, 18, 21}
MANDATORY_COVERAGE_MIN_TURN = 15

EXPLICIT_DENIAL_PATTERNS = (
    "not really",
    "pretty close to normal",
    "about normal",
    "about the same",
    "has been fine",
    "hasnt really been an issue",
    "hasn't really been an issue",
    "nothing different there",
    "not much change there",
    "no real change",
    "havent noticed much change",
    "haven't noticed much change",
    "that part feels normal",
)

CONTRASTIVE_REPLY_PATTERNS = (
    "more than",
    "more toward",
    "a bit of both",
    "both show up",
    "both are in there",
    "more as",
)

SELF_WORTH_CLUSTER_ITEM_IDS = (7, 8, 14)
SELF_WORTH_ROTATION_ORDER = {
    7: (8, 14),
    8: (14, 7),
    14: (8, 7),
}


def _priority_from_gain(expected_gain: float) -> float:
    gain = max(0.0, float(expected_gain))
    return max(0.0, min(1.0, gain / (gain + 2.0)))


def _thread_value(thread: Any, key: str, default: Any) -> Any:
    if isinstance(thread, dict):
        return thread.get(key, default)
    return getattr(thread, key, default)


def _current_thread_state(state: AgentState) -> Dict[str, Any]:
    thread = state.get("conversation_thread")
    return {
        "active": bool(_thread_value(thread, "active", False)),
        "route": str(_thread_value(thread, "route", "cognitive") or "cognitive"),
        "module_id": int(_thread_value(thread, "module_id", 0) or 0),
        "source_item_id": int(_thread_value(thread, "source_item_id", 0) or 0),
        "question_count": int(_thread_value(thread, "question_count", 0) or 0),
        "denial_streak": int(_thread_value(thread, "denial_streak", 0) or 0),
        "last_question_kind": str(_thread_value(thread, "last_question_kind", "") or ""),
        "timeframe_introduced": bool(_thread_value(thread, "timeframe_introduced", False)),
        "anchor_text": str(_thread_value(thread, "anchor_text", "") or ""),
        "exit_reason": str(_thread_value(thread, "exit_reason", "") or ""),
    }


def _latest_persona_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if str(msg.get("role", "")).strip().lower() == "assistant":
            return str(msg.get("content", "") or "").strip()
    return ""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _is_explicit_denial_reply(text: str) -> bool:
    lowered = _normalize_text(text)
    if not lowered:
        return False
    return any(pattern in lowered for pattern in EXPLICIT_DENIAL_PATTERNS)


def _has_contrastive_reply(text: str) -> bool:
    lowered = _normalize_text(text)
    if not lowered:
        return False
    return any(pattern in lowered for pattern in CONTRASTIVE_REPLY_PATTERNS)


def _anchor_text_for_item(state: AgentState, item_id: int) -> str:
    latest_evidence = list(state.get("latest_turn_evidence", []))
    for row in latest_evidence:
        try:
            row_item_id = int(getattr(row, "item_id", 0) or 0)
        except (TypeError, ValueError):
            row_item_id = 0
        if row_item_id != int(item_id):
            continue
        text = str(getattr(row, "evidence_text", "") or "").strip()
        if text:
            return text
    latest_message = _latest_persona_message(state)
    if not latest_message:
        return ""
    sentence = re.split(r"[.!?]", latest_message, maxsplit=1)[0].strip()
    return " ".join(sentence.split()[:14]).strip(" ,;:")


def _route_history_target_items(row: Any) -> List[int]:
    if isinstance(row, dict):
        raw_items = row.get("target_items", []) or []
    else:
        raw_items = getattr(row, "target_items", []) or []
    target_items: List[int] = []
    for raw_item in raw_items:
        try:
            item_id = int(raw_item)
        except (TypeError, ValueError):
            continue
        if 1 <= item_id <= 21 and item_id not in target_items:
            target_items.append(item_id)
    return target_items


def _route_history_string(row: Any, field: str, default: str = "") -> str:
    if isinstance(row, dict):
        value = row.get(field, default)
    else:
        value = getattr(row, field, default)
    text = str(value or "").strip().lower()
    return text or str(default or "").strip().lower()


def _belief_support(item_beliefs: Dict[int, Any], item_id: int) -> int:
    belief = item_beliefs.get(int(item_id))
    try:
        return int(getattr(belief, "support_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _belief_entropy(item_beliefs: Dict[int, Any], item_id: int) -> float:
    belief = item_beliefs.get(int(item_id))
    try:
        return float(getattr(belief, "entropy", 2.0) or 2.0)
    except (TypeError, ValueError):
        return 2.0


def _belief_expected(item_beliefs: Dict[int, Any], item_id: int) -> float:
    belief = item_beliefs.get(int(item_id))
    try:
        return float(getattr(belief, "expected_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _recent_target_counts(route_history: Sequence[Any], window: int = RECENT_COUNT_WINDOW) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in list(route_history)[-window:]:
        for item_id in _route_history_target_items(row):
            counts[item_id] = int(counts.get(item_id, 0)) + 1
    return counts


def _route_for_module(module_id: int) -> str:
    for route_name, module_ids in NODE_ALLOWED_MODULES.items():
        if int(module_id) in module_ids:
            return str(route_name)
    return "cognitive"


def _resolved_module_for_history_row(row: Any, item_beliefs: Dict[int, Any]) -> int:
    target_items = _route_history_target_items(row)
    if not target_items:
        return 0
    route = _route_history_string(row, "chosen_node", "cognitive")
    try:
        return int(choose_target_module(route, target_items, item_beliefs))
    except Exception:
        return 0


def _recent_module_counts(route_history: Sequence[Any], item_beliefs: Dict[int, Any], window: int = RECENT_COUNT_WINDOW) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in list(route_history)[-window:]:
        module_id = _resolved_module_for_history_row(row, item_beliefs)
        if module_id > 0:
            counts[module_id] = int(counts.get(module_id, 0)) + 1
    return counts


def _module_unresolved_ratio(module_id: int, item_beliefs: Dict[int, Any]) -> float:
    module_items = list(MODULE_TO_ITEMS.get(int(module_id), []))
    if not module_items:
        return 1.0
    unresolved = sum(1 for item_id in module_items if _belief_support(item_beliefs, item_id) <= 0)
    return float(unresolved) / float(len(module_items))


def _module_supported_count(module_id: int, item_beliefs: Dict[int, Any]) -> int:
    return sum(1 for item_id in MODULE_TO_ITEMS.get(int(module_id), []) if _belief_support(item_beliefs, item_id) > 0)


def _coerce_trace_int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    items: List[int] = []
    for raw_item in value:
        try:
            item_id = int(raw_item)
        except (TypeError, ValueError):
            continue
        if 1 <= item_id <= 21 and item_id not in items:
            items.append(item_id)
    return items


def _latest_extract_trace(state: AgentState) -> Dict[str, Any]:
    turn_trace = dict(state.get("turn_trace", {}))
    extract_trace = turn_trace.get("extract_evidence")
    if not isinstance(extract_trace, dict):
        extract_trace = turn_trace.get("extract_likelihoods", {})
    if not isinstance(extract_trace, dict):
        extract_trace = {}
    return dict(extract_trace)


def _latest_productive_context(state: AgentState, item_beliefs: Dict[int, Any]) -> Dict[str, Any]:
    turn_trace = dict(state.get("turn_trace", {}))
    belief_trace = turn_trace.get("update_beliefs")
    if not isinstance(belief_trace, dict):
        belief_trace = turn_trace.get("belief_update", {})
    if not isinstance(belief_trace, dict):
        belief_trace = {}

    specialist_trace = turn_trace.get("specialist", {})
    if not isinstance(specialist_trace, dict):
        specialist_trace = {}

    updated_item_ids = _coerce_trace_int_list(belief_trace.get("updated_item_ids", []))
    support_increments_count = 0
    try:
        support_increments_count = int(belief_trace.get("support_increments_count", 0) or 0)
    except (TypeError, ValueError):
        support_increments_count = 0
    productive_turn = bool(support_increments_count > 0 or updated_item_ids)

    route = str(
        specialist_trace.get("node", state.get("active_node", "cognitive")) or state.get("active_node", "cognitive")
    ).strip().lower()
    if route not in {"cognitive", "somatic", "risk"}:
        route = "cognitive"

    target_item_id = 0
    try:
        target_item_id = int(specialist_trace.get("target_item_id", 0) or 0)
    except (TypeError, ValueError):
        target_item_id = 0
    if not (1 <= target_item_id <= 21):
        route_history = list(state.get("route_history", []))
        if route_history:
            targets = _route_history_target_items(route_history[-1])
            if targets:
                target_item_id = int(targets[0])
    if route == "risk":
        target_item_id = 9
    if not (1 <= target_item_id <= 21):
        target_item_id = 2

    target_module_id = 0
    try:
        target_module_id = int(specialist_trace.get("target_module_id", 0) or 0)
    except (TypeError, ValueError):
        target_module_id = 0
    if route == "risk":
        target_module_id = 9
    elif target_module_id not in modules_for_item(target_item_id):
        target_module_id = int(choose_target_module(route, [target_item_id], item_beliefs))

    supported_updated_item_ids = [
        int(item_id)
        for item_id in updated_item_ids
        if _belief_support(item_beliefs, int(item_id)) > 0
    ]
    if supported_updated_item_ids:
        target_item_id = max(
            supported_updated_item_ids,
            key=lambda item_id: (
                float(_belief_expected(item_beliefs, item_id)),
                int(_belief_support(item_beliefs, item_id)),
                -int(item_id),
            ),
        )
        if int(target_item_id) == 9:
            route = "risk"
            target_module_id = 9
        else:
            module_candidates = modules_for_item(target_item_id)
            if module_candidates:
                route = _route_for_module(int(module_candidates[0]))
                target_module_id = int(choose_target_module(route, [target_item_id], item_beliefs))

    return {
        "productive_turn": productive_turn,
        "updated_item_ids": updated_item_ids,
        "support_increments_count": support_increments_count,
        "route": route,
        "target_item_id": target_item_id,
        "target_module_id": target_module_id,
    }


def _thread_seed_context(state: AgentState, item_beliefs: Dict[int, Any], productive_context: Dict[str, Any]) -> Dict[str, Any]:
    latest_turn_evidence = list(state.get("latest_turn_evidence", []))
    candidate_item_ids: List[int] = []
    for row in latest_turn_evidence:
        try:
            item_id = int(getattr(row, "item_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if 1 <= item_id <= 21 and item_id not in candidate_item_ids:
            candidate_item_ids.append(item_id)

    for item_id in productive_context.get("updated_item_ids", []):
        try:
            candidate = int(item_id)
        except (TypeError, ValueError):
            continue
        if 1 <= candidate <= 21 and candidate not in candidate_item_ids:
            candidate_item_ids.append(candidate)

    if not candidate_item_ids and 1 <= int(productive_context.get("target_item_id", 0) or 0) <= 21:
        candidate_item_ids.append(int(productive_context["target_item_id"]))

    if not candidate_item_ids:
        return {
            "source_item_id": 0,
            "module_id": 0,
            "route": str(productive_context.get("route", "cognitive") or "cognitive"),
            "anchor_text": "",
        }

    source_item_id = max(
        candidate_item_ids,
        key=lambda item_id: (
            int(_belief_support(item_beliefs, int(item_id))) > 0,
            float(_belief_expected(item_beliefs, int(item_id))),
            int(_belief_support(item_beliefs, int(item_id))),
            -int(item_id),
        ),
    )
    if int(source_item_id) == 9:
        return {
            "source_item_id": 9,
            "module_id": 9,
            "route": "risk",
            "anchor_text": _anchor_text_for_item(state, 9),
        }

    module_candidates = modules_for_item(source_item_id)
    if module_candidates:
        route = _route_for_module(int(module_candidates[0]))
        module_id = int(choose_target_module(route, [source_item_id], item_beliefs))
    else:
        route = str(productive_context.get("route", "cognitive") or "cognitive")
        module_id = int(productive_context.get("target_module_id", 0) or 0)

    return {
        "source_item_id": int(source_item_id),
        "module_id": int(module_id),
        "route": str(route),
        "anchor_text": _anchor_text_for_item(state, int(source_item_id)),
    }


def _recent_followup_used_for_item(
    route_history: Sequence[Any],
    *,
    item_id: int,
    module_id: int,
    item_beliefs: Dict[int, Any],
    window: int = FOLLOWUP_WINDOW,
) -> bool:
    for row in list(route_history)[-window:]:
        policy = _route_history_string(row, "policy", "")
        if policy not in FOLLOWUP_POLICIES:
            continue
        target_items = _route_history_target_items(row)
        if int(item_id) in target_items:
            return True
        if _resolved_module_for_history_row(row, item_beliefs) == int(module_id):
            return True
    return False


def _recent_followup_used_for_module(
    route_history: Sequence[Any],
    *,
    module_id: int,
    item_beliefs: Dict[int, Any],
    window: int = FOLLOWUP_WINDOW,
) -> bool:
    for row in list(route_history)[-window:]:
        policy = _route_history_string(row, "policy", "")
        if policy not in FOLLOWUP_POLICIES:
            continue
        if _resolved_module_for_history_row(row, item_beliefs) == int(module_id):
            return True
    return False


def _recent_risk_attempted(route_history: Sequence[Any], *, window: int = RISK_REENTRY_LOOKBACK) -> bool:
    for row in list(route_history)[-window:]:
        if 9 in _route_history_target_items(row):
            return True
        if _route_history_string(row, "chosen_node", "") == "risk":
            return True
    return False


def _core_module_support_count(item_beliefs: Dict[int, Any]) -> int:
    core_items = set(MODULE_TO_ITEMS[1]) | set(MODULE_TO_ITEMS[3])
    return sum(1 for item_id in core_items if _belief_support(item_beliefs, item_id) > 0)


def _risk_reentry_reason(state: AgentState, *, item_beliefs: Dict[int, Any], route_history: Sequence[Any]) -> str:
    if _belief_support(item_beliefs, 9) > 0:
        return ""
    if _recent_risk_attempted(route_history):
        return ""
    if bool(state.get("risk_flag", False)):
        return "risk_flag"

    metrics = state.get("metrics")
    total_expected_bdi = 0.0
    try:
        total_expected_bdi = float(getattr(metrics, "total_expected_bdi", 0.0) or 0.0)
    except (TypeError, ValueError):
        total_expected_bdi = 0.0
    if total_expected_bdi >= RISK_REENTRY_TOTAL_BDI_THRESHOLD:
        return "high_expected_bdi"

    if _core_module_support_count(item_beliefs) >= RISK_REENTRY_CORE_SUPPORT_THRESHOLD:
        return "supported_core_modules"
    return ""


def _risk_signal_recent(state: AgentState, *, item_beliefs: Dict[int, Any], productive_context: Dict[str, Any]) -> bool:
    if _belief_support(item_beliefs, 9) > 0:
        return True
    if 9 in set(productive_context.get("updated_item_ids", [])):
        return True

    extract_trace = _latest_extract_trace(state)
    supported_item_ids = _coerce_trace_int_list(extract_trace.get("detail_supported_item_ids", []))
    supported_item_ids.extend(
        item_id
        for item_id in _coerce_trace_int_list(extract_trace.get("opportunistic_supported_item_ids", []))
        if item_id not in supported_item_ids
    )
    if 9 in supported_item_ids:
        return True
    try:
        dropped_by_item9 = int(extract_trace.get("detail_supported_rows_dropped_by_item9", 0) or 0)
    except (TypeError, ValueError):
        dropped_by_item9 = 0
    return dropped_by_item9 > 0


def _ever_targeted_item(route_history: Sequence[Any], item_id: int) -> bool:
    for row in route_history:
        if int(item_id) in _route_history_target_items(row):
            return True
    return False


def _module3_self_worth_recovery_candidate(
    state: AgentState,
    *,
    item_beliefs: Dict[int, Any],
    route_history: Sequence[Any],
    recent_item_counts: Dict[int, int],
    recent_module_counts: Dict[int, int],
    ig_estimates: Dict[int, float],
) -> Tuple[int, float, str, int, List[Dict[str, Any]], int, int] | None:
    context = _latest_productive_context(state, item_beliefs)
    if bool(context["productive_turn"]) or str(context["route"]) == "risk":
        return None

    source_item_id = int(context["target_item_id"] or 0)
    source_module_id = int(context["target_module_id"] or 0)
    if source_module_id != 3 or source_item_id not in SELF_WORTH_CLUSTER_ITEM_IDS:
        return None

    extract_trace = _latest_extract_trace(state)
    supported_item_ids = _coerce_trace_int_list(extract_trace.get("detail_supported_item_ids", []))
    supported_item_ids.extend(
        item_id
        for item_id in _coerce_trace_int_list(extract_trace.get("opportunistic_supported_item_ids", []))
        if item_id not in supported_item_ids
    )
    if any(item_id in SELF_WORTH_CLUSTER_ITEM_IDS for item_id in supported_item_ids):
        return None

    ordered_candidates = list(SELF_WORTH_ROTATION_ORDER.get(source_item_id, (8, 14)))
    candidate_rows: List[Tuple[int, int, float, float, bool]] = []
    for item_id in ordered_candidates:
        if _belief_support(item_beliefs, item_id) > 0:
            continue
        recent_count = int(recent_item_counts.get(item_id, 0))
        entropy = float(_belief_entropy(item_beliefs, item_id))
        ig_estimate = float(ig_estimates.get(item_id, entropy))
        never_targeted = not _ever_targeted_item(route_history, item_id)
        candidate_rows.append((item_id, recent_count, entropy, ig_estimate, never_targeted))

    if not candidate_rows:
        return None

    candidate_rows.sort(
        key=lambda row: (
            not bool(row[4]),
            row[1],
            -row[2],
            -row[3],
            ordered_candidates.index(int(row[0])),
        )
    )
    target_item_id, recent_count, entropy, ig_estimate, never_targeted = candidate_rows[0]
    score = float(ig_estimate) + float(entropy) + (0.45 if never_targeted else 0.20) - (0.10 * float(recent_count))
    ranking = [
        _candidate_entry(
            item_id=item_id,
            route="cognitive",
            module_id=3,
            score=(candidate_ig + candidate_entropy + (0.45 if candidate_never_targeted else 0.20) - (0.10 * candidate_recent_count)),
            item_beliefs=item_beliefs,
            ig_estimates=ig_estimates,
            recent_item_counts=recent_item_counts,
            recent_module_counts=recent_module_counts,
            score_components={
                "ig_estimate": candidate_ig,
                "entropy": candidate_entropy,
                "self_worth_recovery_bonus": 0.45 if candidate_never_targeted else 0.20,
                "recent_item_penalty": 0.10 * float(candidate_recent_count),
            },
        )
        for item_id, candidate_recent_count, candidate_entropy, candidate_ig, candidate_never_targeted in candidate_rows
    ]
    rationale = (
        f"module-3 self-worth recovery after unproductive item {source_item_id} turn; "
        f"rotate to uncovered sibling item {target_item_id}"
    )
    return int(target_item_id), float(score), rationale, 3, ranking, int(source_item_id), 3


def _mandatory_coverage_candidate(
    *,
    turn_index: int,
    item_beliefs: Dict[int, Any],
    route_history: Sequence[Any],
    ig_estimates: Dict[int, float],
    recent_item_counts: Dict[int, int],
    recent_module_counts: Dict[int, int],
) -> Tuple[int, float, str, int, List[Dict[str, Any]], int, int] | None:
    if turn_index < MANDATORY_COVERAGE_MIN_TURN:
        return None
    uncovered = [
        item_id for item_id in sorted(MANDATORY_COVERAGE_ITEMS)
        if _belief_support(item_beliefs, item_id) <= 0
    ]
    if not uncovered:
        return None
    already_targeted = set()
    for row in list(route_history):
        for item_id in _route_history_target_items(row):
            already_targeted.add(item_id)
    candidates = [item_id for item_id in uncovered if item_id not in already_targeted]
    if not candidates:
        return None
    target_item_id = candidates[0]
    module_candidates = modules_for_item(target_item_id)
    route = _route_for_module(module_candidates[0]) if module_candidates else "cognitive"
    module_id = int(choose_target_module(route, [target_item_id], item_beliefs))
    ig_estimate = float(ig_estimates.get(target_item_id, _belief_entropy(item_beliefs, target_item_id)))
    entropy = float(_belief_entropy(item_beliefs, target_item_id))
    score = ig_estimate + entropy + GLOBAL_UNCOVERED_BONUS
    score_components = {
        "ig_estimate": ig_estimate,
        "entropy": entropy,
        "uncovered_bonus": GLOBAL_UNCOVERED_BONUS,
        "mandatory_coverage": 1.0,
    }
    ranking = [
        _candidate_entry(
            item_id=target_item_id,
            route=route,
            module_id=module_id,
            score=score,
            item_beliefs=item_beliefs,
            ig_estimates=ig_estimates,
            recent_item_counts=recent_item_counts,
            recent_module_counts=recent_module_counts,
            score_components=score_components,
        )
    ]
    rationale = f"mandatory coverage: item {target_item_id} unobserved after turn {MANDATORY_COVERAGE_MIN_TURN}"
    return target_item_id, score, rationale, module_id, ranking, 0, 0


def _module_saturation_penalty(module_id: int, item_beliefs: Dict[int, Any]) -> float:
    supported_count = _module_supported_count(module_id, item_beliefs)
    unresolved_ratio = _module_unresolved_ratio(module_id, item_beliefs)
    if (
        supported_count < MODULE_SATURATION_MIN_SUPPORTED
        or unresolved_ratio > MODULE_SATURATION_MAX_UNRESOLVED_RATIO
    ):
        return 0.0

    support_bonus = max(0.0, 0.15 * float(supported_count - MODULE_SATURATION_MIN_SUPPORTED))
    ratio_bonus = max(0.0, (MODULE_SATURATION_MAX_UNRESOLVED_RATIO - unresolved_ratio) * 0.5)
    return 0.75 + support_bonus + ratio_bonus


def _candidate_entry(
    *,
    item_id: int,
    route: str,
    module_id: int,
    score: float,
    item_beliefs: Dict[int, Any],
    ig_estimates: Dict[int, float],
    recent_item_counts: Dict[int, int],
    recent_module_counts: Dict[int, int],
    score_components: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    return {
        "item_id": int(item_id),
        "route": str(route),
        "module_id": int(module_id),
        "module_name": MODULE_NAMES.get(int(module_id), f"Module {int(module_id)}"),
        "score": round(float(score), 4),
        "support_count": int(_belief_support(item_beliefs, item_id)),
        "expected_score": round(float(_belief_expected(item_beliefs, item_id)), 4),
        "entropy": round(float(_belief_entropy(item_beliefs, item_id)), 4),
        "ig_estimate": round(float(ig_estimates.get(int(item_id), _belief_entropy(item_beliefs, item_id))), 4),
        "recent_item_target_count": int(recent_item_counts.get(int(item_id), 0)),
        "recent_module_target_count": int(recent_module_counts.get(int(module_id), 0)),
        "score_components": {
            str(key): round(float(value), 4)
            for key, value in dict(score_components or {}).items()
        },
    }


def _same_item_followup_candidate(
    state: AgentState,
    *,
    item_beliefs: Dict[int, Any],
    route_history: Sequence[Any],
    recent_item_counts: Dict[int, int],
    recent_module_counts: Dict[int, int],
    ig_estimates: Dict[int, float],
) -> Tuple[int, float, str, int, List[Dict[str, Any]], int, int] | None:
    context = _latest_productive_context(state, item_beliefs)
    if not context["productive_turn"] or context["route"] == "risk":
        return None

    item_id = int(context["target_item_id"])
    module_id = int(context["target_module_id"])
    if item_id not in set(context["updated_item_ids"]):
        return None
    if _belief_support(item_beliefs, item_id) <= 0:
        return None

    expected_score = _belief_expected(item_beliefs, item_id)
    if expected_score < SAME_ITEM_EXPECTED_MIN or expected_score > SAME_ITEM_EXPECTED_MAX:
        return None

    if _recent_followup_used_for_item(
        route_history,
        item_id=item_id,
        module_id=module_id,
        item_beliefs=item_beliefs,
    ):
        return None

    route = str(context["route"])
    ig_estimate = float(ig_estimates.get(item_id, _belief_entropy(item_beliefs, item_id)))
    entropy = float(_belief_entropy(item_beliefs, item_id))
    score = ig_estimate + entropy
    score_components = {
        "ig_estimate": ig_estimate,
        "entropy": entropy,
    }
    ranking = [
        _candidate_entry(
            item_id=item_id,
            route=route,
            module_id=module_id,
            score=score,
            item_beliefs=item_beliefs,
            ig_estimates=ig_estimates,
            recent_item_counts=recent_item_counts,
            recent_module_counts=recent_module_counts,
            score_components=score_components,
        )
    ]
    rationale = (
        f"evidence follow-up on item {item_id}: first support landed and severity remains unresolved"
    )
    return item_id, score, rationale, module_id, ranking, item_id, module_id


def _same_module_followup_candidate(
    state: AgentState,
    *,
    item_beliefs: Dict[int, Any],
    route_history: Sequence[Any],
    recent_item_counts: Dict[int, int],
    recent_module_counts: Dict[int, int],
    ig_estimates: Dict[int, float],
) -> Tuple[int, float, str, int, List[Dict[str, Any]], int, int] | None:
    context = _latest_productive_context(state, item_beliefs)
    if not context["productive_turn"] or context["route"] == "risk":
        return None

    module_id = int(context["target_module_id"])
    updated_item_ids = set(context["updated_item_ids"])
    updated_supported_in_module = [
        item_id
        for item_id in updated_item_ids
        if int(item_id) in MODULE_TO_ITEMS.get(module_id, []) and _belief_support(item_beliefs, int(item_id)) > 0
    ]
    if not updated_supported_in_module:
        return None

    source_item_id = max(
        updated_supported_in_module,
        key=lambda item_id: (float(_belief_expected(item_beliefs, item_id)), -int(item_id)),
    )
    if _belief_expected(item_beliefs, source_item_id) < 1.0:
        return None

    if _recent_followup_used_for_module(route_history, module_id=module_id, item_beliefs=item_beliefs):
        return None

    sibling_candidates: List[Tuple[float, int]] = []
    for item_id in MODULE_TO_ITEMS.get(module_id, []):
        if int(item_id) == int(source_item_id):
            continue
        if _belief_support(item_beliefs, int(item_id)) > 0:
            continue
        entropy = float(_belief_entropy(item_beliefs, int(item_id)))
        recent_item_count = int(recent_item_counts.get(int(item_id), 0))
        sibling_candidates.append((entropy, recent_item_count, int(item_id)))

    if not sibling_candidates:
        return None

    sibling_candidates.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
    top_entropy, top_recent_count, target_item_id = sibling_candidates[0]
    route = str(context["route"])
    ranking = [
        _candidate_entry(
            item_id=item_id,
            route=route,
            module_id=module_id,
            score=entropy,
            item_beliefs=item_beliefs,
            ig_estimates=ig_estimates,
            recent_item_counts=recent_item_counts,
            recent_module_counts=recent_module_counts,
            score_components={
                "entropy": entropy,
                "recent_item_penalty": float(recent_item_count),
            },
        )
        for entropy, recent_item_count, item_id in sibling_candidates[:5]
    ]
    rationale = (
        f"evidence follow-up in module {module_id}: supported item {source_item_id} suggests probing unresolved sibling"
    )
    score = float(top_entropy) - (0.10 * float(top_recent_count))
    return target_item_id, score, rationale, module_id, ranking, source_item_id, module_id


def _global_ranked_candidates(
    state: AgentState,
    *,
    item_beliefs: Dict[int, Any],
    route_history: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metrics = state.get("metrics")
    ig_estimates = dict(getattr(metrics, "last_ig_estimates", {}) or {})
    recent_item_counts = _recent_target_counts(route_history)
    recent_module_counts = _recent_module_counts(route_history, item_beliefs)
    productive_context = _latest_productive_context(state, item_beliefs)
    base_risk_recent_attempted = _recent_risk_attempted(route_history)
    base_risk_reentry_reason = _risk_reentry_reason(
        state,
        item_beliefs=item_beliefs,
        route_history=route_history,
    )
    base_risk_reentry_eligible = bool(base_risk_reentry_reason)
    risk_signal_recent = _risk_signal_recent(state, item_beliefs=item_beliefs, productive_context=productive_context)
    risk_recent_attempted = _recent_risk_attempted(route_history)
    risk_reentry_reason = _risk_reentry_reason(state, item_beliefs=item_beliefs, route_history=route_history)
    risk_reentry_eligible = bool(risk_reentry_reason)

    ranked: List[Dict[str, Any]] = []
    any_risk_dampener_applied = False
    for item_id in range(1, 22):
        module_candidates = modules_for_item(item_id)
        if not module_candidates:
            continue

        ig_estimate = float(ig_estimates.get(item_id, _belief_entropy(item_beliefs, item_id)))
        entropy = float(_belief_entropy(item_beliefs, item_id))
        support_count = int(_belief_support(item_beliefs, item_id))
        item_penalty = ITEM_REPETITION_PENALTY * float(recent_item_counts.get(item_id, 0))
        uncovered_bonus = GLOBAL_UNCOVERED_BONUS if support_count == 0 else 0.0

        best_entry: Dict[str, Any] | None = None
        for module_id in module_candidates:
            route = _route_for_module(module_id)
            unresolved_ratio = _module_unresolved_ratio(module_id, item_beliefs)
            module_penalty = MODULE_REPETITION_PENALTY * float(recent_module_counts.get(module_id, 0))
            module_saturation_penalty = _module_saturation_penalty(module_id, item_beliefs)
            risk_dampener = 0.0
            risk_reentry_bonus = 0.0
            if int(item_id) == 9:
                if risk_signal_recent:
                    risk_dampener = 0.0
                elif risk_reentry_eligible:
                    risk_reentry_bonus = RISK_REENTRY_BONUS
                else:
                    risk_dampener = RISK_GLOBAL_DAMPENER
                    any_risk_dampener_applied = True
            score = (
                (GLOBAL_IG_WEIGHT * ig_estimate)
                + (GLOBAL_ENTROPY_WEIGHT * entropy)
                + uncovered_bonus
                + (GLOBAL_MODULE_UNRESOLVED_WEIGHT * unresolved_ratio)
                - item_penalty
                - module_penalty
                - module_saturation_penalty
                - risk_dampener
                + risk_reentry_bonus
            )
            score_components = {
                "ig_estimate": GLOBAL_IG_WEIGHT * ig_estimate,
                "entropy": GLOBAL_ENTROPY_WEIGHT * entropy,
                "uncovered_bonus": uncovered_bonus,
                "module_unresolved": GLOBAL_MODULE_UNRESOLVED_WEIGHT * unresolved_ratio,
                "recent_item_penalty": item_penalty,
                "recent_module_penalty": module_penalty,
                "module_saturation_penalty": module_saturation_penalty,
                "risk_dampener": risk_dampener,
                "risk_reentry_bonus": risk_reentry_bonus,
            }
            entry = _candidate_entry(
                item_id=item_id,
                route=route,
                module_id=module_id,
                score=score,
                item_beliefs=item_beliefs,
                ig_estimates=ig_estimates,
                recent_item_counts=recent_item_counts,
                recent_module_counts=recent_module_counts,
                score_components=score_components,
            )
            entry["module_unresolved_ratio"] = round(float(unresolved_ratio), 4)
            entry["module_saturation_penalty"] = round(float(module_saturation_penalty), 4)
            entry["risk_dampener_applied"] = bool(risk_dampener > 0.0)
            entry["risk_reentry_bonus"] = round(float(risk_reentry_bonus), 4)
            if best_entry is None or (
                float(entry["score"]),
                -int(entry["item_id"]),
                -int(entry["module_id"]),
            ) > (
                float(best_entry["score"]),
                -int(best_entry["item_id"]),
                -int(best_entry["module_id"]),
            ):
                best_entry = entry

        if best_entry is not None:
            ranked.append(best_entry)

    ranked.sort(
        key=lambda entry: (
            float(entry["score"]),
            -int(entry["support_count"]),
            -int(entry["item_id"]),
            -int(entry["module_id"]),
        ),
        reverse=True,
    )
    return ranked, {
        "risk_dampener_applied": bool(any_risk_dampener_applied),
        "risk_reentry_eligible": bool(risk_reentry_eligible),
        "risk_reentry_reason": str(risk_reentry_reason),
        "risk_recent_attempted": bool(risk_recent_attempted),
    }


def target_selector(state: AgentState) -> Dict:
    turn_index = int(state.get("turn_index", 0))
    has_new_persona_input = bool(state.get("has_new_persona_input", False))
    item_beliefs = dict(state.get("item_beliefs", {}))
    route_history = list(state.get("route_history", []))
    current_thread = _current_thread_state(state)
    metrics = state.get("metrics")
    ig_estimates = dict(getattr(metrics, "last_ig_estimates", {}) or {})
    recent_item_counts = _recent_target_counts(route_history)
    recent_module_counts = _recent_module_counts(route_history, item_beliefs)
    productive_context = _latest_productive_context(state, item_beliefs)
    thread_seed = _thread_seed_context(state, item_beliefs, productive_context)
    latest_persona_text = _latest_persona_message(state)
    latest_reply_denial = _is_explicit_denial_reply(latest_persona_text)
    latest_reply_contrastive = _has_contrastive_reply(latest_persona_text)
    base_risk_recent_attempted = _recent_risk_attempted(route_history)
    base_risk_reentry_reason = _risk_reentry_reason(
        state,
        item_beliefs=item_beliefs,
        route_history=route_history,
    )
    base_risk_reentry_eligible = bool(base_risk_reentry_reason)
    question_kind = "topic_open"
    timeframe_mode = "introduce"
    thread_turn_index = 0
    anchor_text = ""
    thread_exit_reason = ""
    thread_source_item_id = 0
    forced_risk_interrupt = False
    module3_self_worth_recovery_applied = False
    module3_self_worth_recovery_eligible = False
    module3_self_worth_source_item_id = 0

    if turn_index == 0 and not has_new_persona_input:
        target_item_id = 2
        route = "cognitive"
        style = "opening"
        question_kind = "opening"
        timeframe_mode = "introduce"
        policy = "opening_bootstrap"
        selected_module_id = int(choose_target_module(route, [target_item_id], item_beliefs))
        expected_gain = 0.0
        rationale = "opening bootstrap"
        ranking_top_candidates = [
            _candidate_entry(
                item_id=target_item_id,
                route=route,
                module_id=selected_module_id,
                score=expected_gain,
                item_beliefs=item_beliefs,
                ig_estimates=ig_estimates,
                recent_item_counts=recent_item_counts,
                recent_module_counts=recent_module_counts,
                score_components={"bootstrap": 0.0},
            )
        ]
        followup_source_item_id = 0
        followup_source_module_id = 0
        same_module_followup_eligible = False
        same_item_followup_eligible = False
        global_risk_dampener_applied = False
        risk_reentry_eligible = bool(base_risk_reentry_eligible)
        risk_reentry_reason = str(base_risk_reentry_reason)
        risk_recent_attempted = bool(base_risk_recent_attempted)
    else:
        style = "gentle_probe"
        followup_source_item_id = 0
        followup_source_module_id = 0
        global_risk_dampener_applied = False
        risk_reentry_eligible = bool(base_risk_reentry_eligible)
        risk_reentry_reason = str(base_risk_reentry_reason)
        risk_recent_attempted = bool(base_risk_recent_attempted)
        same_module_candidate = None
        same_item_candidate = None
        module3_self_worth_candidate = None
        same_module_followup_eligible = False
        same_item_followup_eligible = False

        if has_new_persona_input:
            same_module_candidate = _same_module_followup_candidate(
                state,
                item_beliefs=item_beliefs,
                route_history=route_history,
                recent_item_counts=recent_item_counts,
                recent_module_counts=recent_module_counts,
                ig_estimates=ig_estimates,
            )
            same_item_candidate = _same_item_followup_candidate(
                state,
                item_beliefs=item_beliefs,
                route_history=route_history,
                recent_item_counts=recent_item_counts,
                recent_module_counts=recent_module_counts,
                ig_estimates=ig_estimates,
            )
            module3_self_worth_candidate = _module3_self_worth_recovery_candidate(
                state,
                item_beliefs=item_beliefs,
                route_history=route_history,
                recent_item_counts=recent_item_counts,
                recent_module_counts=recent_module_counts,
                ig_estimates=ig_estimates,
            )
        same_module_followup_eligible = same_module_candidate is not None
        same_item_followup_eligible = same_item_candidate is not None
        module3_self_worth_recovery_eligible = module3_self_worth_candidate is not None

        active_thread = bool(current_thread["active"])
        thread_denial_streak = int(current_thread["denial_streak"])

        if has_new_persona_input and base_risk_reentry_eligible and not base_risk_recent_attempted:
            if bool(state.get("risk_flag", False)) or int(thread_seed["source_item_id"]) == 9 or int(productive_context["target_item_id"]) == 9:
                forced_risk_interrupt = True

        if active_thread:
            if forced_risk_interrupt:
                thread_exit_reason = "risk_interruption"
                active_thread = False
            elif int(current_thread["question_count"]) >= THREAD_MAX_QUESTIONS:
                thread_exit_reason = "thread_budget_reached"
                active_thread = False
            else:
                if latest_reply_denial:
                    thread_denial_streak += 1
                else:
                    thread_denial_streak = 0
                if thread_denial_streak >= THREAD_EXIT_DENIALS:
                    thread_exit_reason = "consecutive_denials"
                    active_thread = False
                elif not productive_context["productive_turn"] and not latest_reply_contrastive:
                    thread_exit_reason = "uninformative_no_update"
                    active_thread = False

        if forced_risk_interrupt:
            target_item_id = 9
            route = "risk"
            selected_module_id = 9
            expected_gain = float(ig_estimates.get(9, _belief_entropy(item_beliefs, 9)))
            rationale = "risk interruption from recent signal"
            policy = "evidence_weighted_global"
            question_kind = "risk_check"
            timeframe_mode = "introduce"
            style = "gentle_probe"
            ranking_top_candidates = [
                _candidate_entry(
                    item_id=9,
                    route="risk",
                    module_id=9,
                    score=expected_gain,
                    item_beliefs=item_beliefs,
                    ig_estimates=ig_estimates,
                    recent_item_counts=recent_item_counts,
                    recent_module_counts=recent_module_counts,
                    score_components={"risk_interrupt": expected_gain},
                )
            ]
            anchor_text = thread_seed["anchor_text"]
            thread_turn_index = 0
        elif active_thread:
            thread_source_item_id = int(current_thread["source_item_id"] or thread_seed["source_item_id"] or productive_context["target_item_id"])
            if latest_reply_contrastive and same_module_candidate is not None:
                (
                    target_item_id,
                    expected_gain,
                    rationale,
                    selected_module_id,
                    ranking_top_candidates,
                    followup_source_item_id,
                    followup_source_module_id,
                ) = same_module_candidate
                route = str(current_thread["route"] or productive_context["route"])
                policy = "conversation_contrastive_pivot"
                question_kind = "contrastive_pivot"
                style = "gentle_probe"
                timeframe_mode = "carry"
                thread_turn_index = int(current_thread["question_count"]) + 1
            elif same_item_candidate is not None:
                (
                    target_item_id,
                    expected_gain,
                    rationale,
                    selected_module_id,
                    ranking_top_candidates,
                    followup_source_item_id,
                    followup_source_module_id,
                ) = same_item_candidate
                route = str(current_thread["route"] or productive_context["route"])
                policy = "evidence_followup_same_item"
                question_kind = "same_item_followup"
                style = "clarify_frequency" if int(current_thread["question_count"]) <= 1 else "functional_impact"
                timeframe_mode = "clarify" if style == "clarify_frequency" else "carry"
                thread_turn_index = int(current_thread["question_count"]) + 1
            elif same_module_candidate is not None:
                (
                    target_item_id,
                    expected_gain,
                    rationale,
                    selected_module_id,
                    ranking_top_candidates,
                    followup_source_item_id,
                    followup_source_module_id,
                ) = same_module_candidate
                route = str(current_thread["route"] or productive_context["route"])
                policy = "evidence_followup_same_module"
                question_kind = "same_module_followup"
                style = "functional_impact" if int(current_thread["question_count"]) >= 2 else "gentle_probe"
                timeframe_mode = "carry"
                thread_turn_index = int(current_thread["question_count"]) + 1
            else:
                active_thread = False
                thread_exit_reason = thread_exit_reason or "thread_exhausted"

            if active_thread:
                if int(followup_source_item_id or 0) > 0:
                    thread_source_item_id = int(followup_source_item_id)
                anchor_text = thread_seed["anchor_text"] or current_thread["anchor_text"]

        if not forced_risk_interrupt and not active_thread:
            start_new_thread = (
                has_new_persona_input
                and productive_context["productive_turn"]
                and int(thread_seed["source_item_id"]) > 0
                and str(thread_seed["route"]) != "risk"
                and not bool(thread_exit_reason)
            )
            if start_new_thread:
                target_item_id = int(thread_seed["source_item_id"])
                route = str(thread_seed["route"])
                selected_module_id = int(thread_seed["module_id"])
                expected_gain = float(ig_estimates.get(target_item_id, _belief_entropy(item_beliefs, target_item_id)))
                rationale = f"conversation thread start from item {target_item_id}"
                policy = "conversation_topic_open"
                question_kind = "topic_open"
                timeframe_mode = "introduce"
                style = "gentle_probe"
                thread_turn_index = 1
                thread_source_item_id = int(target_item_id)
                anchor_text = str(thread_seed["anchor_text"])
                ranking_top_candidates = [
                    _candidate_entry(
                        item_id=target_item_id,
                        route=route,
                        module_id=selected_module_id,
                        score=expected_gain,
                        item_beliefs=item_beliefs,
                        ig_estimates=ig_estimates,
                        recent_item_counts=recent_item_counts,
                        recent_module_counts=recent_module_counts,
                        score_components={"thread_start": expected_gain},
                    )
                ]
            elif module3_self_worth_candidate is not None:
                (
                    target_item_id,
                    expected_gain,
                    rationale,
                    selected_module_id,
                    ranking_top_candidates,
                    followup_source_item_id,
                    followup_source_module_id,
                ) = module3_self_worth_candidate
                route = "cognitive"
                policy = "module3_self_worth_recovery"
                question_kind = "topic_open"
                timeframe_mode = "introduce"
                style = "gentle_probe"
                thread_turn_index = 1
                thread_source_item_id = int(target_item_id)
                anchor_text = thread_seed["anchor_text"] or _anchor_text_for_item(state, int(target_item_id))
                module3_self_worth_recovery_applied = True
                module3_self_worth_source_item_id = int(followup_source_item_id or 0)
            else:
                mandatory_candidate = _mandatory_coverage_candidate(
                    turn_index=turn_index,
                    item_beliefs=item_beliefs,
                    route_history=route_history,
                    ig_estimates=ig_estimates,
                    recent_item_counts=recent_item_counts,
                    recent_module_counts=recent_module_counts,
                )
                if mandatory_candidate is not None:
                    (
                        target_item_id,
                        expected_gain,
                        rationale,
                        selected_module_id,
                        ranking_top_candidates,
                        followup_source_item_id,
                        followup_source_module_id,
                    ) = mandatory_candidate
                    route = _route_for_module(selected_module_id)
                    policy = "mandatory_coverage"
                else:
                    ranked_candidates, global_meta = _global_ranked_candidates(
                        state,
                        item_beliefs=item_beliefs,
                        route_history=route_history,
                    )
                    global_risk_dampener_applied = bool(global_meta["risk_dampener_applied"])
                    risk_reentry_eligible = bool(global_meta["risk_reentry_eligible"])
                    risk_reentry_reason = str(global_meta["risk_reentry_reason"])
                    risk_recent_attempted = bool(global_meta["risk_recent_attempted"])
                    selected = ranked_candidates[0] if ranked_candidates else {
                        "item_id": 2,
                        "route": "cognitive",
                        "module_id": 2,
                        "score": 0.0,
                    }
                    target_item_id = int(selected["item_id"])
                    route = str(selected["route"])
                    selected_module_id = int(selected["module_id"])
                    expected_gain = float(selected["score"])
                    rationale = "evidence-weighted global ranking"
                    ranking_top_candidates = ranked_candidates[:5]
                    policy = "evidence_weighted_global"

                if str(route) == "risk":
                    question_kind = "risk_check"
                    timeframe_mode = "introduce"
                    style = "gentle_probe"
                    thread_turn_index = 0
                    thread_source_item_id = 0
                    anchor_text = thread_seed["anchor_text"]
                else:
                    question_kind = "topic_open"
                    timeframe_mode = "introduce"
                    style = "gentle_probe"
                    thread_turn_index = 1
                    thread_source_item_id = int(target_item_id)
                    anchor_text = thread_seed["anchor_text"] or _anchor_text_for_item(state, int(target_item_id))

    next_action = NextAction(
        target_item_id=target_item_id,
        route=route,
        style=style,
        mode="normal",
        directness=(
            "direct"
            if question_kind == "risk_check" or (question_kind == "same_item_followup" and style == "clarify_frequency")
            else "indirect"
        ),
        priority=_priority_from_gain(expected_gain),
        rationale=rationale,
        question_kind=question_kind,
        thread_turn_index=int(thread_turn_index),
        thread_module_id=int(selected_module_id),
        thread_source_item_id=int(thread_source_item_id or target_item_id),
        timeframe_mode=timeframe_mode,
        anchor_text=str(anchor_text or ""),
    )

    decision = RouteDecision(
        turn=max(1, turn_index),
        chosen_node=route,
        policy=policy,
        reason=rationale,
        target_items=[target_item_id],
        expected_gain=float(expected_gain),
    )

    selected_item_recent_count = int(recent_item_counts.get(int(target_item_id), 0))
    selected_module_recent_count = int(recent_module_counts.get(int(selected_module_id), 0))
    selected_entry = ranking_top_candidates[0] if ranking_top_candidates else {}
    selected_score_components = dict(selected_entry.get("score_components", {}) or {})
    selected_module_saturation_penalty = float(selected_score_components.get("module_saturation_penalty", 0.0) or 0.0)
    selected_risk_dampener_applied = bool(selected_score_components.get("risk_dampener", 0.0))

    turn_trace = dict(state.get("turn_trace", {}))
    selector_trace = {
        "turn": max(1, turn_index),
        "policy": policy,
        "chosen_node": route,
        "reason": rationale,
        "target_items": [target_item_id],
        "expected_gain": round(float(expected_gain), 4),
        "selected_module_id": int(selected_module_id),
        "selected_module_name": MODULE_NAMES.get(int(selected_module_id), f"Module {int(selected_module_id)}"),
        "conversation_thread_active": bool(question_kind in {"topic_open", "same_item_followup", "same_module_followup", "contrastive_pivot"}),
        "question_kind": question_kind,
        "thread_turn_index": int(thread_turn_index),
        "thread_exit_reason": thread_exit_reason,
        "timeframe_mode": timeframe_mode,
        "anchor_text": str(anchor_text or ""),
        "productive_turn": bool(productive_context["productive_turn"]),
        "updated_item_ids": list(productive_context["updated_item_ids"]),
        "same_module_followup_eligible": bool(same_module_followup_eligible),
        "same_item_followup_eligible": bool(same_item_followup_eligible),
        "module3_self_worth_recovery_eligible": bool(module3_self_worth_recovery_eligible),
        "module3_self_worth_recovery_applied": bool(module3_self_worth_recovery_applied),
        "module3_self_worth_source_item_id": int(module3_self_worth_source_item_id),
        "followup_source_item_id": int(followup_source_item_id),
        "followup_source_module_id": int(followup_source_module_id),
        "recent_item_target_count": selected_item_recent_count,
        "recent_module_target_count": selected_module_recent_count,
        "module_saturation_penalty": round(float(selected_module_saturation_penalty), 4),
        "risk_dampener_applied": bool(global_risk_dampener_applied or selected_risk_dampener_applied),
        "risk_reentry_eligible": bool(risk_reentry_eligible),
        "risk_reentry_reason": str(risk_reentry_reason),
        "risk_recent_attempted": bool(risk_recent_attempted),
        "selected_score_components": selected_score_components,
        "ranking_top_candidates": ranking_top_candidates,
    }
    turn_trace["target_selector"] = selector_trace
    turn_trace["supervisor"] = selector_trace

    debug = (
        f"Target selector -> {route}; target_item={target_item_id}; module={selected_module_id}; "
        f"style={style}; question_kind={question_kind}; policy={policy}; gain={expected_gain:.2f}"
    )

    conversation_thread = ConversationThreadState(
        active=bool(question_kind in {"topic_open", "same_item_followup", "same_module_followup", "contrastive_pivot"}),
        route=route if route in {"somatic", "cognitive", "risk"} else "cognitive",
        module_id=int(selected_module_id),
        source_item_id=int(thread_source_item_id or target_item_id),
        question_count=int(thread_turn_index if question_kind != "opening" else 0),
        denial_streak=0 if not latest_reply_denial else min(THREAD_EXIT_DENIALS, int(current_thread["denial_streak"]) + 1),
        last_question_kind=question_kind,
        timeframe_introduced=bool(timeframe_mode in {"introduce", "clarify"} or current_thread["timeframe_introduced"]),
        anchor_text=str(anchor_text or ""),
        exit_reason=str(thread_exit_reason),
    )
    if route == "risk" or question_kind == "opening":
        conversation_thread.active = False
        conversation_thread.question_count = 0
        conversation_thread.denial_streak = 0

    return {
        "next_action": next_action,
        "conversation_thread": conversation_thread,
        "next_node": route,
        "active_node": route,
        "route_history": [decision],
        "route_debug": debug,
        "turn_trace": turn_trace,
    }
