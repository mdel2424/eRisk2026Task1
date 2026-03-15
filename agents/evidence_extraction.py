from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Sequence, Tuple

from agents.evidence_lexicon import LEXICAL_EVIDENCE_CUES
from core.bdi_modules import MODULE_NAMES, MODULE_TO_ITEMS, choose_target_module
from core.llm import LLMBudgetExceeded, get_extractor_llm
from core.prompts import get_prompt
from core.state import (
    AgentState,
    BDI_ITEM_NAMES,
    EvidenceRecord,
    LikelihoodEvidence,
    SYMPTOM_NAME_TO_ITEM,
    bump_failure_counter,
)

EXTRACTOR_KEY_ALIASES = {
    "item": "item_id",
    "id": "item_id",
    "bdi_item": "item_id",
    "bdi_item_id": "item_id",
    "symptom": "symptom_name",
    "label": "symptom_name",
    "conf": "confidence",
    "certainty": "confidence",
    "score": "intensity",
    "severity": "intensity",
    "rationale": "reason",
    "quote": "anchor_quote",
    "anchor": "anchor_quote",
    "evidence_quote": "anchor_quote",
    "is_supported": "supported",
    "support_flag": "supported",
}

METHOD_WEIGHT_HINTS = {
    "llm_extractor": 1.00,
    "llm_opportunistic": 0.75,
    "llm_salvage": 0.60,
    "lexical_fallback": 0.45,
    "lexical_prefilter": 0.40,
}

ITEM1_STRONG_PATTERNS = (
    re.compile(r"\bfeel(?:ing)?\s+sad\b"),
    re.compile(r"\bfelt\s+sad\b"),
    re.compile(r"\blow\s+mood\b"),
    re.compile(r"\bfeeling\s+down\b"),
    re.compile(r"\btearful\b"),
    re.compile(r"\bcr(?:y|ies|ied|ying)\b"),
    re.compile(r"\bemotionally\s+flat\b"),
    re.compile(r"\bemotionally\s+numb\b"),
    re.compile(r"\bfeel\b.{0,20}\bsad\b"),
    re.compile(r"\bheaviness\b"),
    re.compile(r"\bweighed\s+down\b"),
    re.compile(r"\bheartbroken\b"),
)

ITEM1_WEAK_PATTERNS = (
    re.compile(r"\bbeen\s+down\b"),
    re.compile(r"\bmood\s+down\b"),
    re.compile(r"\bkind\s+of\s+flat\b"),
    re.compile(r"\bfeels?\s+flat\b"),
    re.compile(r"\bfeels?\s+numb\b"),
    re.compile(r"\bdown\s+lately\b"),
    re.compile(r"\bfeeling\s+(?:low|blue|grey|gray)\b"),
    re.compile(r"\bfeeling\s+empty\b"),
)

PRECISION_GUARD_METHODS = {"lexical_fallback", "llm_salvage"}
PRECISION_GUARD_ITEM9 = {9}
PRECISION_GUARD_MODULE1 = {4, 10, 12, 17}
PRECISION_GUARD_MODULE3 = {3, 5, 6, 7, 8, 14}
PRECISION_GUARD_MODULE4 = {13, 19}
ALL_BDI_ITEM_IDS = tuple(range(1, 22))

ITEM9_DIRECT_PATTERNS = (
    re.compile(r"\bbetter\s+off\s+dead\b"),
    re.compile(r"\bkill\s+myself\b"),
    re.compile(r"\bself[\s\-]?harm\b"),
    re.compile(r"\bhurt\s+myself\b"),
    re.compile(r"\bwish\s+i\s+wasn'?t\s+here\b"),
    re.compile(r"\bdon'?t\s+want\s+to\s+wake\s+up\b"),
    re.compile(r"\bnot\s+wake\s+up\b"),
)

ITEM9_PASSIVE_RISK_PATTERNS = (
    re.compile(r"\b(?:wish|wished)\s+i\s+could\s+(?:just\s+)?disappear\b"),
    re.compile(r"\bit(?:'d| would)\s+be\s+easier\s+if\s+i\s+(?:just\s+)?disappeared\b"),
    re.compile(r"\b(?:rather|would\s+almost\s+)\b.{0,24}\bnot\s+be\s+here\s+anymore\b"),
    re.compile(r"\bbetter\s+if\s+i\s+(?:just\s+)?disappeared\b"),
    re.compile(r"\bnot\s+being\s+here\s+feels\s+easier\b"),
    re.compile(r"\bit\s+feels\s+easier\s+not\s+to\s+be\s+here\b"),
    re.compile(r"\bnot\s+being\s+here\s+would\s+be\s+easier\b"),
)

MODULE1_STRONG_PATTERNS = (
    re.compile(r"\b(?:enjoy|enjoying|enjoyed|pleasure|reward(?:ing)?|fun|look\s+forward)\b"),
    re.compile(r"\b(?:withdrawing|keeping\s+to\s+myself|avoid(?:ing)?\s+people|pulling\s+away|ignoring\s+messages|cancel(?:ing)?\s+plans)\b"),
    re.compile(r"\b(?:tearful|cr(?:y|ies|ied|ying))\b"),
    re.compile(r"\b(?:irritable|irritability|short\s+fuse|snapp(?:y|ish)|easily\s+annoyed)\b"),
)

MODULE1_WEAK_PATTERNS = (
    re.compile(r"\b(?:going\s+through\s+the\s+motions|feel(?:s|ing)?\s+like\s+a\s+chore)\b"),
    re.compile(r"\b(?:autopilot|disconnected|checked\s+out|not\s+really\s+being\s+present|hollow)\b"),
)

MODULE3_STRONG_PATTERNS = (
    re.compile(r"\b(?:guilty|guilt|ashamed|shame|my\s+fault|blame\s+myself|regret)\b"),
    re.compile(r"\b(?:worthless|useless|burden|not\s+enough|not\s+good\s+enough)\b"),
    re.compile(r"\b(?:failure|failed|hate\s+myself|dislike\s+myself|self[\s\-]?critical|beat\s+myself\s+up)\b"),
    re.compile(r"\bmy\s+own\s+fault\b"),
    re.compile(r"\bguilty\s+feelings?\b"),
    re.compile(r"\bdon'?t\s+measure\s+up\b"),
    re.compile(r"\bdo\s+not\s+measure\s+up\b"),
    re.compile(r"\bdon'?t\s+matter\b"),
    re.compile(r"\bdo\s+not\s+matter\b"),
    re.compile(r"\bdon'?t\s+contribute\b"),
    re.compile(r"\bdo\s+not\s+contribute\b"),
    re.compile(r"\bdon'?t\s+like\s+the\s+person\s+i(?:'ve| have)\s+been\b"),
    re.compile(r"\bdislike\s+who\s+i(?:'ve| have)\s+become\b"),
)

MODULE3_WEAK_PATTERNS = (
    re.compile(r"\b(?:stress(?:ed)?|pressure|frustrat(?:ed|ing)|behind|catch[\s\-]?up|overwhelmed)\b"),
    re.compile(r"\b(?:second[\s\-]?guess(?:ing)?|self[\s\-]?doubt)\b"),
)

MODULE3_SOFT_SUPPORT_PATTERNS: Dict[int, Tuple[re.Pattern[str], ...]] = {
    5: (
        re.compile(r"\bguilt(?:y)?\b"),
        re.compile(r"\bguilty\s+feelings?\b"),
        re.compile(r"\bguilt\s+just\s+shows\s+up\b"),
        re.compile(r"\bmy\s+own\s+fault\b"),
        re.compile(r"\bmy\s+fault\b"),
        re.compile(r"\bblame\s+myself\b"),
        re.compile(r"\bashamed\b"),
        re.compile(r"\bregret\b"),
    ),
    7: (
        re.compile(r"\bhate\s+myself\b"),
        re.compile(r"\bdislike\s+myself\b"),
        re.compile(r"\bdon'?t\s+like\s+myself\b"),
    ),
    8: (
        re.compile(r"\bself[\s\-]?critical\b"),
        re.compile(r"\bbeat\s+myself\s+up\b"),
        re.compile(r"\bdon'?t\s+measure\s+up\b"),
        re.compile(r"\bdo\s+not\s+measure\s+up\b"),
        re.compile(r"\bhard\s+on\s+myself\b"),
        re.compile(r"\bwhat\s+i(?:'ve| have)\s+done\s+wrong\b"),
    ),
    14: (
        re.compile(r"\bworthless\b"),
        re.compile(r"\buseless\b"),
        re.compile(r"\bfeel\s+like\s+a\s+burden\b"),
        re.compile(r"\b(?:a\s+)?burden\b"),
        re.compile(r"\bnot\s+enough\b"),
        re.compile(r"\bdon'?t\s+matter\b"),
        re.compile(r"\bdo\s+not\s+matter\b"),
        re.compile(r"\bdon'?t\s+contribute\b"),
        re.compile(r"\bdo\s+not\s+contribute\b"),
    ),
}

ITEM14_LATENT_SUPPORT_PATTERNS = (
    re.compile(r"\bfeel\s+worthless\b"),
    re.compile(r"\bworthless\b"),
    re.compile(r"\buseless\b"),
    re.compile(r"\bfeel\s+like\s+a\s+burden\b"),
    re.compile(r"\bi\s+am\s+a\s+burden\b"),
    re.compile(r"\bi'?m\s+a\s+burden\b"),
    re.compile(r"\bdon'?t\s+matter\b"),
    re.compile(r"\bdo\s+not\s+matter\b"),
    re.compile(r"\bdon'?t\s+contribute\b"),
    re.compile(r"\bdo\s+not\s+contribute\b"),
    re.compile(r"\bfeel\s+like\s+a\s+failure\b"),
    re.compile(r"\bi\s+am\s+a\s+failure\b"),
    re.compile(r"\bi'?m\s+a\s+failure\b"),
    re.compile(r"\bdon'?t\s+like\s+who\s+i\s+am\b"),
    re.compile(r"\bdo\s+not\s+like\s+who\s+i\s+am\b"),
    re.compile(r"\bdon'?t\s+like\s+the\s+person\s+i(?:'ve| have)\s+been\b"),
    re.compile(r"\bdo\s+not\s+like\s+the\s+person\s+i(?:'ve| have)\s+been\b"),
    re.compile(r"\bdislike\s+who\s+i(?:'ve| have)\s+become\b"),
    re.compile(r"\bdon'?t\s+measure\s+up\b"),
    re.compile(r"\bdo\s+not\s+measure\s+up\b"),
)

ITEM14_IDENTITY_CHANGE_PATTERNS = (
    re.compile(r"\bdon'?t\s+like\s+who\s+i\s+am\b"),
    re.compile(r"\bdo\s+not\s+like\s+who\s+i\s+am\b"),
    re.compile(r"\bdon'?t\s+like\s+the\s+person\s+i(?:'ve| have)\s+been\b"),
    re.compile(r"\bdo\s+not\s+like\s+the\s+person\s+i(?:'ve| have)\s+been\b"),
    re.compile(r"\bdislike\s+who\s+i(?:'ve| have)\s+become\b"),
)

ITEM14_WORTHLESSNESS_PATTERNS = (
    re.compile(r"\bworthless\b"),
    re.compile(r"\buseless\b"),
    re.compile(r"\bfeel\s+like\s+a\s+burden\b"),
    re.compile(r"\b(?:a\s+)?burden\b"),
    re.compile(r"\bdon'?t\s+matter\b"),
    re.compile(r"\bdo\s+not\s+matter\b"),
    re.compile(r"\bdon'?t\s+contribute\b"),
    re.compile(r"\bdo\s+not\s+contribute\b"),
)

ITEM21_SEXUAL_QUESTION_PATTERNS = (
    re.compile(r"\breduced\s+interest\s+in\s+sexual\s+activity\b"),
    re.compile(r"\blittle\s+or\s+no\s+interest\s+in\s+sexual\s+activity\b"),
    re.compile(r"\bsexual\s+activity\b"),
    re.compile(r"\binterest\s+in\s+sexual\b"),
    re.compile(r"\binterest\s+in\s+sex\b"),
    re.compile(r"\blittle\s+or\s+no\s+interest\s+in\s+sexual\b"),
    re.compile(r"\bhow\s+interested\s+you\s+usually\s+are\b"),
    re.compile(r"\bcompared\s+with\s+your\s+usual\s+level\b"),
)

ITEM21_MILD_DIRECT_PATTERNS = (
    re.compile(r"\bthat\s+side\s+of\s+things\s+is\s+a\s+little\s+lower\s+than\s+usual\b"),
    re.compile(r"\bthat\s+side\s+of\s+things\s+is\s+a\s+bit\s+lower\s+than\s+usual\b"),
    re.compile(r"\bside\s+of\s+things\s+is\s+a\s+little\s+lower\s+than\s+usual\b"),
    re.compile(r"\ba\s+little\s+lower\s+than\s+usual\b"),
    re.compile(r"\ba\s+bit\s+lower\s+than\s+usual\b"),
    re.compile(r"\bless\s+interested\s+than\s+usual\b"),
    re.compile(r"\bnot\s+as\s+interested\s+as\s+usual\b"),
    re.compile(r"\blittle\s+less\s+interested\b"),
    re.compile(r"\binterest\s+is\s+a\s+little\s+lower\b"),
    re.compile(r"\binterest\s+is\s+down\s+a\s+bit\b"),
    re.compile(r"\breduced\s+interest\b"),
    re.compile(r"\blower\s+than\s+usual\b"),
)

ITEM21_DIRECT_DENIAL_PATTERNS = (
    re.compile(r"\bthat\s+side\s+of\s+things\s+has\s+been\s+fine\b"),
    re.compile(r"\bthat\s+side\s+of\s+things\s+is\s+fine\b"),
    re.compile(r"\bnot\s+really\s+a\s+problem\b"),
    re.compile(r"\bhasn'?t\s+really\s+been\s+an\s+issue\b"),
    re.compile(r"\bhas\s+not\s+really\s+been\s+an\s+issue\b"),
    re.compile(r"\bnothing\s+different\s+there\b"),
    re.compile(r"\bokay\s+honestly\b"),
)

ITEM18_APPETITE_SIGNAL_PATTERNS = (
    re.compile(r"\bappetite\b"),
    re.compile(r"\beat(?:ing)?\b"),
    re.compile(r"\bmeals?\b"),
    re.compile(r"\bhungry\b"),
    re.compile(r"\bfood\b"),
)

ITEM18_APPETITE_BASELINE_CHANGE_PATTERNS = (
    re.compile(r"\bmuch\s+less\b"),
    re.compile(r"\ba\s+little\s+less\b"),
    re.compile(r"\bmuch\s+more\b"),
    re.compile(r"\ba\s+little\s+more\b"),
    re.compile(r"\bless\s+than\s+usual\b"),
    re.compile(r"\bmore\s+than\s+usual\b"),
    re.compile(r"\blower\s+than\s+usual\b"),
    re.compile(r"\bincreased\b"),
    re.compile(r"\bdecreased\b"),
    re.compile(r"\bchange(?:d)?\b"),
)

ITEM18_APPETITE_DYSFUNCTION_PATTERNS = (
    re.compile(r"\bnot\s+eating\s+at\s+all\b"),
    re.compile(r"\b(?:barely|hardly)\s+eat(?:ing)?\b"),
    re.compile(r"\bgrabbing\s+junk\b"),
    re.compile(r"\beat\s+because\s+i\s+have\s+to\b"),
    re.compile(r"\bmeals?\s+feel\s+more\s+like\s+a\s+chore\b"),
)

MODULE4_STRONG_PATTERNS = (
    re.compile(r"\b(?:can'?t\s+focus|lose\s+focus|concentrat(?:e|ing|ion)|can'?t\s+think\s+clearly)\b"),
    re.compile(r"\b(?:indecisive|indecision|can'?t\s+decide|hard\s+to\s+decide|stuck\s+choosing|decision(?:s)?\s+feel\s+hard)\b"),
)

MODULE4_WEAK_PATTERNS = (
    re.compile(r"\b(?:brain\s+fog|foggy|mind\s+(?:was\s+)?racing|overwhelmed|scattered|all\s+over\s+the\s+place)\b"),
    re.compile(r"\b(?:zoning\s+out|mind\s+is\s+always\s+elsewhere)\b"),
)

CLEAR_NO_SYMPTOM_PATTERNS = (
    re.compile(
        r"\b(?:not\s+really|nothing(?:'s|\s+is)?\s+(?:different|changed|new|unusual)|"
        r"no\s+real\s+change|same\s+as\s+usual|about\s+normal|pretty\s+normal|unchanged)\b"
    ),
    re.compile(r"\b(?:been\s+fine|doing\s+fine|it'?s\s+fine|okay\s+overall|all\s+right)\b"),
    re.compile(r"\b(?:nothing\s+much|not\s+much\s+to\s+say|same\s+old|usual\s+stuff)\b"),
    re.compile(r"\b(?:that|this)\s+side\s+of\s+things\s+has\s+been\s+(?:fine|okay|ok|normal)\b"),
    re.compile(r"\b(?:no\s+issue\s+there|not\s+(?:much\s+of\s+)?an?\s+issue|hasn'?t\s+really\s+been\s+(?:much\s+of\s+)?an?\s+issue)\b"),
    re.compile(r"\b(?:nothing\s+out\s+of\s+the\s+ordinary|pretty\s+much\s+the\s+same|about\s+the\s+same)\b"),
    re.compile(r"\b(?:that|this|it)\s+has(?:n'?t)?\s+been\s+(?:a\s+)?problem\b"),
    re.compile(r"\b(?:[a-z]+(?:\s+[a-z]+){0,3}\s+)?hasn'?t\s+really\s+been\s+(?:much\s+of\s+)?(?:an?\s+)?(?:issue|problem)\b"),
)

PURE_LOGISTICAL_PATTERNS = (
    re.compile(r"\b(?:just|mostly)?\s*busy(?:\s+lately|\s+with\s+(?:work|school|classes))?\b"),
    re.compile(r"\b(?:work|school|classes|schedule|routine)\s+(?:has|have|is|been)\s+busy\b"),
)

SYMPTOM_SIGNAL_PATTERN = re.compile(
    r"\b(?:sad|down|low|hopeless|empty|numb|guilty|worthless|burden|failure|fault|blame|"
    r"sleep|awake|wake|appetite|eat|eating|hungry|tired|fatigue|energy|focus|concentrat|"
    r"decid|cry|tearful|irritable|agitated|restless|suicid|kill\s+myself|wish\s+i\s+wasn'?t\s+here|"
    r"don'?t\s+want\s+to\s+wake\s+up|not\s+wake\s+up|can'?t\s+get\s+going|getting\s+out\s+of\s+bed)\b"
)

AFFIRMATIVE_CHANGE_PATTERNS = (
    *ITEM9_DIRECT_PATTERNS,
    *MODULE1_STRONG_PATTERNS,
    *MODULE3_STRONG_PATTERNS,
    *MODULE4_STRONG_PATTERNS,
    re.compile(r"\b(?:more|less)\s+than\s+usual\b"),
    re.compile(r"\b(?:worse|harder|struggling|struggle|battle|burden|worthless|guilty)\b"),
    re.compile(r"\b(?:can'?t|get\s+back\s+to\s+sleep|wake\s+up\s+in\s+the\s+middle\s+of\s+the\s+night)\b"),
    re.compile(r"\b(?:takes?\s+so\s+much\s+energy|everything\s+feels\s+heavier|decisions?\s+take\s+me\s+longer)\b"),
)


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))



def _recent_context(state: AgentState, limit: int = 4) -> str:
    turns = state.get("messages", [])[-limit:]
    lines = []
    for msg in turns:
        role = "Detector" if msg.get("role") == "user" else "Persona"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _previous_detector_question(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _extractor_version() -> str:
    return "v2" if os.getenv("PROMPT_VERSION", "v1").strip().lower() == "v2" else "v1"


def _state_field(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _coerce_item_id_int(raw_value: Any, default: int) -> int:
    try:
        item_id = int(raw_value)
    except (TypeError, ValueError):
        return int(default)
    return item_id if 1 <= item_id <= 21 else int(default)


def _extract_target_spec(state: AgentState, node_name: str) -> Dict[str, Any]:
    turn_trace = state.get("turn_trace", {})
    specialist_trace = turn_trace.get("specialist", {}) if isinstance(turn_trace, dict) else {}
    next_action = state.get("next_action")

    route = str(specialist_trace.get("node", node_name) or node_name).strip().lower()
    if route not in {"somatic", "cognitive", "risk"}:
        route = str(node_name).strip().lower()
    if route not in {"somatic", "cognitive", "risk"}:
        route = "cognitive"

    default_item_id = 9 if route == "risk" else 2
    target_item_id = _coerce_item_id_int(
        specialist_trace.get("target_item_id", _state_field(next_action, "target_item_id", default_item_id)),
        default_item_id,
    )
    if route == "risk":
        target_item_id = 9

    target_module_id = specialist_trace.get("target_module_id")
    try:
        target_module_id = int(target_module_id)
    except (TypeError, ValueError):
        target_module_id = 0

    if route == "risk":
        target_module_id = 9
        allowed_item_ids = [9]
    else:
        if target_module_id not in MODULE_TO_ITEMS:
            target_module_id = choose_target_module(
                node_name=route,
                target_items=[target_item_id],
                item_beliefs=state.get("item_beliefs", {}),
            )
        allowed_item_ids = sorted(set([target_item_id] + list(MODULE_TO_ITEMS.get(target_module_id, []))))

    return {
        "route": route,
        "target_item_id": int(target_item_id),
        "target_item_name": BDI_ITEM_NAMES.get(int(target_item_id), f"Item {int(target_item_id)}"),
        "target_module_id": int(target_module_id),
        "target_module_name": MODULE_NAMES.get(int(target_module_id), "General Screening"),
        "allowed_item_ids": list(allowed_item_ids),
    }


def _filter_records_to_allowed_items(
    records: Sequence[EvidenceRecord],
    *,
    allowed_item_ids: Sequence[int],
) -> tuple[List[EvidenceRecord], int]:
    allowed = {int(item_id) for item_id in allowed_item_ids}
    kept: List[EvidenceRecord] = []
    dropped = 0
    for record in records:
        if int(record.item_id) in allowed:
            kept.append(record)
        else:
            dropped += 1
    return kept, dropped


def _filter_items_to_allowed_scope(
    items: Sequence[Dict[str, Any]],
    *,
    allowed_item_ids: Sequence[int],
) -> tuple[List[Dict[str, Any]], int]:
    allowed = {int(item_id) for item_id in allowed_item_ids}
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = _coerce_item_id(item.get("item_id"), str(item.get("symptom_name", "") or ""))
        if item_id is None or int(item_id) not in allowed:
            dropped += 1
            continue
        kept.append(dict(item))
    return kept, dropped


def _coerce_gate_payload(
    parsed: Any,
    *,
    allowed_item_ids: Sequence[int],
) -> Dict[str, Any]:
    allowed = {int(item_id) for item_id in allowed_item_ids}
    if not isinstance(parsed, dict):
        return {
            "target_relevant": False,
            "candidate_item_ids": [],
            "anchor_quote": "",
            "confidence": 0.0,
            "reason": "",
        }

    candidate_item_ids: List[int] = []
    raw_candidates = parsed.get("candidate_item_ids", [])
    if isinstance(raw_candidates, list):
        for raw_item in raw_candidates:
            try:
                item_id = int(raw_item)
            except (TypeError, ValueError):
                continue
            if item_id in allowed and item_id not in candidate_item_ids:
                candidate_item_ids.append(item_id)

    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    target_relevant = bool(parsed.get("target_relevant", False))
    if not target_relevant:
        candidate_item_ids = []

    return {
        "target_relevant": target_relevant,
        "candidate_item_ids": candidate_item_ids[:2],
        "anchor_quote": str(parsed.get("anchor_quote", "") or "").strip(),
        "confidence": _clamp(confidence, 0.0, 1.0),
        "reason": str(parsed.get("reason", "") or "").strip(),
    }


def _coerce_opportunistic_shortlist_payload(
    parsed: Any,
    *,
    scoped_allowed_item_ids: Sequence[int],
) -> Dict[str, Any]:
    scoped_allowed = {int(item_id) for item_id in scoped_allowed_item_ids}
    if not isinstance(parsed, dict):
        return {
            "has_strong_offtarget_signal": False,
            "candidate_item_ids": [],
            "anchor_quote": "",
            "confidence": 0.0,
            "reason": "",
        }

    candidate_item_ids: List[int] = []
    raw_candidates = parsed.get("candidate_item_ids", [])
    if isinstance(raw_candidates, list):
        for raw_item in raw_candidates:
            try:
                item_id = int(raw_item)
            except (TypeError, ValueError):
                continue
            if not (1 <= item_id <= 21):
                continue
            if item_id in scoped_allowed:
                continue
            if item_id not in candidate_item_ids:
                candidate_item_ids.append(item_id)

    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    has_strong_offtarget_signal = bool(parsed.get("has_strong_offtarget_signal", False))
    if not has_strong_offtarget_signal:
        candidate_item_ids = []

    return {
        "has_strong_offtarget_signal": has_strong_offtarget_signal,
        "candidate_item_ids": candidate_item_ids[:4],
        "anchor_quote": str(parsed.get("anchor_quote", "") or "").strip(),
        "confidence": _clamp(confidence, 0.0, 1.0),
        "reason": str(parsed.get("reason", "") or "").strip(),
    }



def _is_clear_no_symptom_reply(latest_message: str, *, previous_question: str = "") -> bool:
    text = str(latest_message or "").strip().lower()
    if not text:
        return True
    explicit_no_signal = any(pattern.search(text) for pattern in CLEAR_NO_SYMPTOM_PATTERNS)
    if explicit_no_signal and not any(pattern.search(text) for pattern in AFFIRMATIVE_CHANGE_PATTERNS):
        return True
    if SYMPTOM_SIGNAL_PATTERN.search(text):
        return False

    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count <= 8 and any(pattern.search(text) for pattern in PURE_LOGISTICAL_PATTERNS):
        return True
    if word_count <= 5 and re.fullmatch(r"(?:yeah[, ]*)?(?:fine|okay|ok|normal|busy)(?:\s+lately|\s+overall)?[.!]?", text):
        return True

    previous_text = str(previous_question or "").strip().lower()
    if previous_text and word_count <= 6 and re.fullmatch(
        r"(?:same|about\s+normal|not\s+really|nothing\s+different|unchanged)[.!]?",
        text,
    ):
        return True
    return False


def _parse_json_payload(raw_text: str) -> Tuple[Any, bool, Dict[str, Any]]:
    text = raw_text.strip()
    if not text:
        return (
            {},
            False,
            {
                "error_kind": "empty_output",
                "error_message": "Extractor output is empty",
                "brace_open": 0,
                "brace_close": 0,
                "bracket_open": 0,
                "bracket_close": 0,
                "double_quote_count": 0,
                "unmatched_double_quote": False,
            },
        )

    def _normalize_quotes(value: str) -> str:
        return (
            value.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )

    def _strip_markdown_fence(value: str) -> str:
        stripped = value.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        if lines and lines[0].strip().lower() in {"json", "application/json"}:
            lines = lines[1:]
        return "\n".join(lines).strip()

    def _cleanup_candidate(value: str) -> str:
        cleaned = _normalize_quotes(_strip_markdown_fence(value))
        cleaned = re.sub(r"^\s*json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return cleaned.strip()

    def _balanced_segments(value: str, open_char: str, close_char: str) -> List[str]:
        segments: List[str] = []
        start = -1
        depth = 0
        for idx, ch in enumerate(value):
            if ch == open_char:
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == close_char and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    segment = value[start : idx + 1].strip()
                    if segment:
                        segments.append(segment)
                    start = -1
        return segments

    cleaned_text = _cleanup_candidate(text)
    candidates: List[str] = [cleaned_text]

    obj_start = cleaned_text.find("{")
    obj_end = cleaned_text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(cleaned_text[obj_start : obj_end + 1])

    arr_start = cleaned_text.find("[")
    arr_end = cleaned_text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidates.append(cleaned_text[arr_start : arr_end + 1])

    for segment in _balanced_segments(cleaned_text, "{", "}"):
        if "item_id" in segment or "evidence" in segment:
            candidates.append(segment)
    for segment in _balanced_segments(cleaned_text, "[", "]"):
        if "item_id" in segment or "evidence" in segment:
            candidates.append(segment)

    def _shape_diagnostics(value: str) -> Dict[str, Any]:
        brace_open = value.count("{")
        brace_close = value.count("}")
        bracket_open = value.count("[")
        bracket_close = value.count("]")
        quote_count = value.count('"')
        unmatched_quote = (quote_count % 2) != 0
        return {
            "brace_open": brace_open,
            "brace_close": brace_close,
            "bracket_open": bracket_open,
            "bracket_close": bracket_close,
            "double_quote_count": quote_count,
            "unmatched_double_quote": unmatched_quote,
        }

    def _infer_error_kind(shape: Dict[str, Any], had_json_error: bool) -> str:
        if shape["brace_open"] > shape["brace_close"]:
            return "missing_closing_brace"
        if shape["brace_close"] > shape["brace_open"]:
            return "extra_closing_brace"
        if shape["bracket_open"] > shape["bracket_close"]:
            return "missing_closing_bracket"
        if shape["bracket_close"] > shape["bracket_open"]:
            return "extra_closing_bracket"
        if bool(shape["unmatched_double_quote"]):
            return "unmatched_quote"
        if had_json_error:
            return "json_decode_error"
        return "no_json_like_payload"

    shape = _shape_diagnostics(cleaned_text)
    first_error: Dict[str, Any] | None = None
    had_json_error = False

    seen: set[str] = set()
    for candidate in candidates:
        normalized = _cleanup_candidate(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError as exc:
            had_json_error = True
            if first_error is None:
                first_error = {
                    "message": str(exc),
                    "line": int(getattr(exc, "lineno", 0) or 0),
                    "column": int(getattr(exc, "colno", 0) or 0),
                    "position": int(getattr(exc, "pos", 0) or 0),
                    "candidate_preview": normalized[:180],
                }
            continue
        if isinstance(payload, (dict, list)):
            return payload, True, {"error_kind": "", "error_message": "", **shape}

    diagnostics: Dict[str, Any] = {
        "error_kind": _infer_error_kind(shape, had_json_error),
        "error_message": str(first_error.get("message", "")) if isinstance(first_error, dict) else "",
        **shape,
    }
    if isinstance(first_error, dict):
        diagnostics["error_line"] = int(first_error.get("line", 0) or 0)
        diagnostics["error_column"] = int(first_error.get("column", 0) or 0)
        diagnostics["error_position"] = int(first_error.get("position", 0) or 0)
        diagnostics["error_candidate_preview"] = str(first_error.get("candidate_preview", "") or "")
    return {}, False, diagnostics



def _number_in_range(value: Any, low: float, high: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return low <= numeric <= high


def _normalize_item_keys(
    item: Dict[str, Any],
    *,
    key_aliases_enabled: bool,
) -> tuple[Dict[str, Any], int]:
    normalized = dict(item)
    alias_hits = 0
    if not key_aliases_enabled:
        return normalized, alias_hits
    for alias, canonical in EXTRACTOR_KEY_ALIASES.items():
        if canonical in normalized:
            continue
        if alias in normalized:
            normalized[canonical] = normalized.get(alias)
            alias_hits += 1
    return normalized, alias_hits


def _coerce_schema_defaults(
    item: Dict[str, Any],
    *,
    strict_schema_coerce: bool,
) -> tuple[Dict[str, Any], int]:
    normalized = dict(item)
    coerce_count = 0
    if not strict_schema_coerce:
        return normalized, coerce_count

    defaults: Dict[str, Any] = {
        "direction": "neutral",
        "intensity": 1.0,
        "confidence": 0.4,
        "evidence_text": "",
        "reason": "schema-coerced extractor output",
    }
    for key, value in defaults.items():
        if key not in normalized or normalized.get(key) in {None, ""}:
            normalized[key] = value
            coerce_count += 1
    return normalized, coerce_count


def _payload_items(parsed: Any) -> tuple[List[Any], int]:
    schema_coerce_used = 0
    items: List[Any] = []
    if isinstance(parsed, dict):
        for key in ("evidence", "items", "records"):
            value = parsed.get(key)
            if isinstance(value, list):
                items = value
                if key != "evidence":
                    schema_coerce_used += 1
                break
        if not items and "item_id" in parsed:
            items = [parsed]
            schema_coerce_used += 1
    elif isinstance(parsed, list):
        items = parsed
    return items, schema_coerce_used


def _payload_scored_items(parsed: Any) -> tuple[List[Any], int]:
    schema_coerce_used = 0
    items: List[Any] = []
    if isinstance(parsed, dict):
        for key in ("scores", "scored_items", "item_scores", "items", "evidence"):
            value = parsed.get(key)
            if isinstance(value, list):
                items = value
                if key != "scores":
                    schema_coerce_used += 1
                break
        if not items and "item_id" in parsed:
            items = [parsed]
            schema_coerce_used += 1
    elif isinstance(parsed, list):
        items = parsed
    return items, schema_coerce_used


def _coerce_supported_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "supported"}


def _coerce_scored_schema_defaults(
    item: Dict[str, Any],
    *,
    strict_schema_coerce: bool,
) -> tuple[Dict[str, Any], int]:
    normalized = dict(item)
    coerce_count = 0
    if not strict_schema_coerce:
        return normalized, coerce_count

    if "supported" not in normalized or normalized.get("supported") in {None, ""}:
        normalized["supported"] = False
        coerce_count += 1
    if "anchor_quote" not in normalized and str(normalized.get("evidence_text", "")).strip():
        normalized["anchor_quote"] = str(normalized.get("evidence_text", "")).strip()
        coerce_count += 1

    defaults: Dict[str, Any] = {
        "confidence": 0.0,
        "intensity": 0.0,
        "anchor_quote": "",
        "reason": "schema-coerced scorer output",
    }
    for key, value in defaults.items():
        if key not in normalized or normalized.get(key) in {None, ""}:
            normalized[key] = value
            coerce_count += 1
    return normalized, coerce_count

def _sentence_for_cue(text: str, cue: str) -> str:
    chunks = [part.strip() for part in text.replace("!", ".").replace("?", ".").split(".")]
    lower_cue = cue.lower()
    for chunk in chunks:
        if lower_cue in chunk.lower():
            return chunk
    return text[:220].strip()


def _normalize_evidence_text_for_id(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _evidence_id(record: EvidenceRecord) -> str:
    normalized_text = _normalize_evidence_text_for_id(record.evidence_text)
    base = f"{int(record.item_id)}|{str(record.direction)}|{normalized_text}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def _has_explicit_sadness_signal(text: str) -> str:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return "none"
    for pattern in ITEM1_STRONG_PATTERNS:
        if pattern.search(normalized):
            return "strong"
    for pattern in ITEM1_WEAK_PATTERNS:
        if pattern.search(normalized):
            return "weak"
    return "none"


def _is_item1_llm_candidate(record: EvidenceRecord) -> bool:
    if int(record.item_id) != 1:
        return False
    if str(record.direction).strip().lower() != "increase":
        return False
    return str(record.method).strip().lower() in {"llm_extractor", "llm_salvage", "llm_opportunistic"}


def _apply_item1_gate(
    record: EvidenceRecord,
    *,
    latest_message: str,
    strict_gate: bool,
    weak_max_conf: float,
    weak_max_intensity: float,
) -> tuple[EvidenceRecord | None, str]:
    signal = _has_explicit_sadness_signal(f"{record.evidence_text}\n{latest_message}")
    if signal == "none":
        if strict_gate:
            return None, "dropped"
        clamped = record.model_copy(
            update={
                "confidence": min(float(record.confidence), weak_max_conf),
                "intensity": min(float(record.intensity), weak_max_intensity),
            }
        )
        if (
            float(clamped.confidence) < float(record.confidence)
            or float(clamped.intensity) < float(record.intensity)
        ):
            return clamped, "soft_clamped"
        return record, "kept"

    if signal == "weak":
        clamped = record.model_copy(
            update={
                "confidence": min(float(record.confidence), weak_max_conf),
                "intensity": min(float(record.intensity), weak_max_intensity),
            }
        )
        if (
            float(clamped.confidence) < float(record.confidence)
            or float(clamped.intensity) < float(record.intensity)
        ):
            return clamped, "soft_clamped"
    return record, "kept"


def _precision_gate_bucket(item_id: int) -> str:
    if item_id in PRECISION_GUARD_ITEM9:
        return "item9"
    if item_id in PRECISION_GUARD_MODULE1:
        return "module1"
    if item_id in PRECISION_GUARD_MODULE3:
        return "module3"
    if item_id in PRECISION_GUARD_MODULE4:
        return "module4"
    return "none"


def _has_any_pattern(text: str, patterns: Tuple[re.Pattern[str], ...]) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in patterns)


def _item9_precision_signal(text: str) -> str:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return "none"
    if any(pattern.search(normalized) for pattern in ITEM9_DIRECT_PATTERNS):
        return "direct"
    if any(pattern.search(normalized) for pattern in ITEM9_PASSIVE_RISK_PATTERNS):
        return "passive_risk"
    return "none"


def _precision_signal_strength(
    text: str,
    *,
    strong_patterns: Tuple[re.Pattern[str], ...],
    weak_patterns: Tuple[re.Pattern[str], ...],
) -> str:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return "none"
    if any(pattern.search(normalized) for pattern in strong_patterns):
        return "strong"
    if any(pattern.search(normalized) for pattern in weak_patterns):
        return "weak"
    return "none"


def _module3_soft_support_item_ids(text: str) -> List[int]:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return []

    supported_item_ids: List[int] = []
    for item_id, patterns in MODULE3_SOFT_SUPPORT_PATTERNS.items():
        if any(pattern.search(normalized) for pattern in patterns):
            supported_item_ids.append(int(item_id))
    return supported_item_ids


def _has_item14_worthlessness_semantics(text: str) -> bool:
    return _has_any_pattern(text, ITEM14_WORTHLESSNESS_PATTERNS)


def _has_item14_latent_support_semantics(text: str) -> bool:
    return _has_any_pattern(text, ITEM14_LATENT_SUPPORT_PATTERNS)


def _has_item14_identity_change_semantics(text: str) -> bool:
    return _has_any_pattern(text, ITEM14_IDENTITY_CHANGE_PATTERNS)


def _is_item21_detector_question(question: str) -> bool:
    return _has_any_pattern(question, ITEM21_SEXUAL_QUESTION_PATTERNS)


def _has_item21_mild_direct_signal(text: str) -> bool:
    return _has_any_pattern(text, ITEM21_MILD_DIRECT_PATTERNS)


def _has_item21_direct_denial(text: str) -> bool:
    return _has_any_pattern(text, ITEM21_DIRECT_DENIAL_PATTERNS)


def _item18_has_direct_change_signal(text: str, *, previous_question: str = "") -> bool:
    if _is_clear_no_symptom_reply(text, previous_question=previous_question):
        return False
    if not _has_any_pattern(text, ITEM18_APPETITE_SIGNAL_PATTERNS):
        return False
    if _has_any_pattern(text, ITEM18_APPETITE_DYSFUNCTION_PATTERNS):
        return True
    return _has_any_pattern(text, ITEM18_APPETITE_BASELINE_CHANGE_PATTERNS)


def _apply_precision_gate(
    record: EvidenceRecord,
    *,
    latest_message: str,
    force_guard: bool = False,
    guard_buckets: set[str] | None = None,
    routed_risk_item9_allowed: bool = False,
) -> tuple[EvidenceRecord | None, str]:
    method = str(record.method or "").strip().lower()
    item_id = int(record.item_id)
    bucket = _precision_gate_bucket(item_id)
    normalized_guard_buckets = (
        {str(bucket_name).strip().lower() for bucket_name in guard_buckets}
        if guard_buckets is not None
        else None
    )
    if normalized_guard_buckets is None:
        if method not in PRECISION_GUARD_METHODS and not force_guard:
            return record, "kept"
    elif bucket not in normalized_guard_buckets:
        return record, "kept"

    if bucket == "none":
        return record, "kept"

    combined_text = f"{record.evidence_text}\n{latest_message}"
    if bucket == "item9":
        item9_signal = _item9_precision_signal(combined_text)
        if item9_signal == "direct":
            return record.model_copy(update={"precision_gate_action": "kept"}), "kept"
        if routed_risk_item9_allowed and item9_signal == "passive_risk":
            return record.model_copy(update={"precision_gate_action": "kept_routed_risk_passive"}), "kept"
        return None, "dropped"

    if bucket == "module1":
        signal = _precision_signal_strength(
            combined_text,
            strong_patterns=MODULE1_STRONG_PATTERNS,
            weak_patterns=MODULE1_WEAK_PATTERNS,
        )
    elif bucket == "module3":
        signal = _precision_signal_strength(
            combined_text,
            strong_patterns=MODULE3_STRONG_PATTERNS,
            weak_patterns=MODULE3_WEAK_PATTERNS,
        )
    else:
        signal = _precision_signal_strength(
            combined_text,
            strong_patterns=MODULE4_STRONG_PATTERNS,
            weak_patterns=MODULE4_WEAK_PATTERNS,
        )

    if signal == "strong":
        return record.model_copy(update={"precision_gate_action": "kept"}), "kept"
    if signal == "weak":
        clamped = record.model_copy(
            update={
                "confidence": min(float(record.confidence), 0.35),
                "intensity": min(float(record.intensity), 1.0),
                "precision_gate_action": "soft_clamped",
                "support_increment_blocked": True,
            }
        )
        return clamped, "soft_clamped"
    return None, "dropped"


def _merge_precision_gate_counts(
    target: Dict[str, Dict[str, int]],
    *,
    item_id: int,
    action: str,
) -> None:
    if action not in {"soft_clamped", "dropped"}:
        return
    key = str(int(item_id))
    if key not in target:
        target[key] = {"soft_clamped": 0, "dropped": 0}
    target[key][action] = int(target[key].get(action, 0) or 0) + 1


def _merge_precision_gate_item_counts(
    target: Dict[str, Dict[str, int]],
    updates: Dict[str, Dict[str, int]],
) -> None:
    for item_id, counts in updates.items():
        if item_id not in target:
            target[item_id] = {"soft_clamped": 0, "dropped": 0}
        target[item_id]["soft_clamped"] += int(counts.get("soft_clamped", 0) or 0)
        target[item_id]["dropped"] += int(counts.get("dropped", 0) or 0)


def _apply_precision_gate_batch(
    records: List[EvidenceRecord],
    *,
    latest_message: str,
    force_guard: bool = False,
    guard_buckets: set[str] | None = None,
    routed_risk_item9_allowed: bool = False,
) -> tuple[List[EvidenceRecord], int, int, Dict[str, Dict[str, int]]]:
    kept_records: List[EvidenceRecord] = []
    dropped_count = 0
    soft_clamped_count = 0
    item_counts: Dict[str, Dict[str, int]] = {}

    for record in records:
        gated_record, action = _apply_precision_gate(
            record,
            latest_message=latest_message,
            force_guard=force_guard,
            guard_buckets=guard_buckets,
            routed_risk_item9_allowed=routed_risk_item9_allowed,
        )
        if action == "dropped":
            dropped_count += 1
            _merge_precision_gate_counts(item_counts, item_id=int(record.item_id), action=action)
            continue
        if action == "soft_clamped":
            soft_clamped_count += 1
            _merge_precision_gate_counts(item_counts, item_id=int(record.item_id), action=action)
        kept_records.append(gated_record if gated_record is not None else record)

    return kept_records, dropped_count, soft_clamped_count, item_counts


def _cue_direction(sentence: str, cue: str) -> str:
    lowered_sentence = str(sentence or "").lower()
    lowered_cue = str(cue or "").lower().strip()
    if not lowered_sentence or not lowered_cue:
        return "increase"

    idx = lowered_sentence.find(lowered_cue)
    if idx < 0:
        return "increase"

    prefix = lowered_sentence[max(0, idx - 96) : idx]
    local = lowered_sentence[max(0, idx - 24) : idx + len(lowered_cue) + 24]

    if re.search(r"\b(?:can(?:not|'t)\s+(?:stop|shake))\b", local):
        return "increase"
    if re.search(r"\bnot\s+only\b", prefix):
        return "increase"

    negation_re = re.compile(
        r"\b(?:no|not|never|without|hardly|rarely|don'?t|didn'?t|haven'?t|hasn'?t|"
        r"won'?t|cannot|can'?t|isn'?t|aren'?t|wasn'?t|weren'?t)\b"
    )
    if negation_re.search(prefix):
        return "decrease"
    return "increase"



def _fallback_evidence_from_text(node_name: str, turn: int, text: str) -> List[EvidenceRecord]:
    lowered = text.lower()
    records: List[EvidenceRecord] = []
    for item_id, cues in LEXICAL_EVIDENCE_CUES.items():
        hit_rows: List[Tuple[str, str, str]] = []
        for cue in cues:
            if cue not in lowered:
                continue
            sentence = _sentence_for_cue(text, cue)
            direction = _cue_direction(sentence, cue)
            hit_rows.append((cue, sentence, direction))
        if not hit_rows:
            continue

        increase_hits = [row for row in hit_rows if row[2] == "increase"]
        decrease_hits = [row for row in hit_rows if row[2] == "decrease"]

        if increase_hits:
            direction = "increase"
            cue, evidence_text, _ = increase_hits[0]
            hit_count = len(increase_hits)
            intensity = min(3.0, 0.90 + (0.30 * hit_count))
            confidence = min(0.82, 0.40 + (0.08 * hit_count))
        elif decrease_hits:
            direction = "decrease"
            cue, evidence_text, _ = decrease_hits[0]
            hit_count = len(decrease_hits)
            intensity = min(3.0, 1.10 + (0.25 * hit_count))
            confidence = min(0.90, 0.55 + (0.08 * hit_count))
        else:
            continue

        records.append(
            EvidenceRecord(
                turn=turn,
                node=node_name if node_name in {"somatic", "cognitive", "risk"} else "cognitive",
                item_id=item_id,
                symptom_name=BDI_ITEM_NAMES.get(item_id, f"Item {item_id}"),
                direction=direction,
                intensity=float(intensity),
                confidence=float(confidence),
                evidence_text=evidence_text,
                reason=f"lexical cue match: {cue} ({direction})",
                method="lexical_fallback",
            )
        )
    records.sort(key=lambda record: (record.confidence, record.intensity), reverse=True)
    return records[:4]



def _coerce_item_id(raw_item_id: Any, raw_symptom_name: str) -> int | None:
    try:
        item_id = int(raw_item_id)
        if 1 <= item_id <= 21:
            return item_id
    except (TypeError, ValueError):
        pass

    symptom = raw_symptom_name.strip().lower()
    if symptom in SYMPTOM_NAME_TO_ITEM:
        return SYMPTOM_NAME_TO_ITEM[symptom]
    return None


def _canonicalize_symptom_name(
    item_id: int,
    raw_symptom_name: str,
) -> tuple[str, bool]:
    canonical = BDI_ITEM_NAMES.get(item_id, f"Item {item_id}")
    incoming = str(raw_symptom_name or "").strip()
    if not incoming:
        return canonical, False
    if incoming.lower() == canonical.lower():
        return canonical, False
    return canonical, True



def _coerce_evidence_record(node_name: str, turn: int, item: Dict, fallback_text: str) -> EvidenceRecord | None:
    raw_symptom_name = str(item.get("symptom_name", "")).strip()
    item_id = _coerce_item_id(item.get("item_id"), raw_symptom_name)
    if item_id is None:
        return None
    symptom_name, _ = _canonicalize_symptom_name(item_id, raw_symptom_name)

    direction = str(item.get("direction", "increase")).strip().lower()
    if direction not in {"increase", "decrease", "neutral"}:
        direction = "increase"

    try:
        intensity = float(item.get("intensity", 0.0))
    except (TypeError, ValueError):
        intensity = 0.0
    intensity = max(0.0, min(3.0, intensity))

    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence_text = str(item.get("evidence_text", "")).strip() or fallback_text[:220]
    reason = str(item.get("reason", "")).strip() or "implicit affective signal"
    method = str(item.get("method", "llm_extractor")).strip() or "llm_extractor"

    return EvidenceRecord(
        turn=turn,
        node=node_name,
        item_id=item_id,
        symptom_name=symptom_name,
        direction=direction,
        intensity=intensity,
        confidence=confidence,
        evidence_text=evidence_text,
        reason=reason,
        method=method,
    )


def _salvage_items_from_text(raw_text: str) -> List[Dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        return []

    line_items: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}

    item_id_re = re.compile(r"\b(?:item[\s_\-]*id|item|id|bdi_item)\b\s*[:=]?\s*(\d{1,2})", re.IGNORECASE)
    symptom_re = re.compile(
        r"\b(?:symptom[\s_\-]*name|symptom|label)\b\s*[:=]?\s*([A-Za-z][A-Za-z \-']+)",
        re.IGNORECASE,
    )
    intensity_re = re.compile(r"\b(?:intensity|score|severity)\b\s*[:=]?\s*([0-3](?:\.\d+)?)", re.IGNORECASE)
    confidence_re = re.compile(
        r"\b(?:confidence|conf|certainty)\b\s*[:=]?\s*((?:0(?:\.\d+)?)|(?:1(?:\.0+)?))",
        re.IGNORECASE,
    )
    direction_re = re.compile(r"\bdirection\b\s*[:=]?\s*(increase|decrease|neutral)", re.IGNORECASE)
    evidence_text_re = re.compile(r"\bevidence[\s_\-]*text\b\s*[:=]?\s*(.+)$", re.IGNORECASE)
    reason_re = re.compile(r"\breason\b\s*[:=]?\s*(.+)$", re.IGNORECASE)

    def _flush() -> None:
        nonlocal current
        if current:
            line_items.append(dict(current))
            current = {}

    candidate_lines = text.splitlines()
    if len(candidate_lines) <= 1:
        candidate_lines = re.split(r"[;|]", text)
    for raw_line in candidate_lines:
        line = raw_line.strip().strip("-*").strip()
        if not line:
            continue

        item_match = item_id_re.search(line)
        if item_match:
            if "item_id" in current:
                _flush()
            current["item_id"] = int(item_match.group(1))

        symptom_match = symptom_re.search(line)
        if symptom_match:
            current["symptom_name"] = symptom_match.group(1).strip().strip('"')

        intensity_match = intensity_re.search(line)
        if intensity_match:
            try:
                current["intensity"] = float(intensity_match.group(1))
            except (TypeError, ValueError):
                pass

        confidence_match = confidence_re.search(line)
        if confidence_match:
            try:
                current["confidence"] = float(confidence_match.group(1))
            except (TypeError, ValueError):
                pass

        direction_match = direction_re.search(line)
        if direction_match:
            current["direction"] = direction_match.group(1).lower()

        evidence_text_match = evidence_text_re.search(line)
        if evidence_text_match:
            current["evidence_text"] = evidence_text_match.group(1).strip().strip('"')

        reason_match = reason_re.search(line)
        if reason_match:
            current["reason"] = reason_match.group(1).strip().strip('"')

    _flush()

    sanitized: List[Dict[str, Any]] = []
    for item in line_items:
        if "item_id" not in item and not str(item.get("symptom_name", "")).strip():
            continue
        item.setdefault("symptom_name", "")
        item.setdefault("direction", "increase")
        item.setdefault("intensity", 1.0)
        item.setdefault("confidence", 0.4)
        item.setdefault("evidence_text", "")
        item.setdefault("reason", "salvaged extractor output")
        item["method"] = "llm_salvage"
        sanitized.append(item)
    return sanitized[:6]


def _records_from_llm_items(
    items: Sequence[Dict[str, Any]],
    *,
    node_name: str,
    turn: int,
    latest_message: str,
    key_aliases_enabled: bool,
    strict_schema_coerce: bool,
    item1_strict_gate: bool,
    item1_weak_max_conf: float,
    item1_weak_max_intensity: float,
    method_override: str | None = None,
    force_guard: bool = False,
    guard_buckets: set[str] | None = None,
    routed_risk_item9_allowed: bool = False,
    min_confidence: float | None = None,
    min_intensity: float | None = None,
    max_records: int | None = None,
) -> tuple[List[EvidenceRecord], Dict[str, Any]]:
    records: List[EvidenceRecord] = []
    stats: Dict[str, Any] = {
        "dropped_unknown": 0,
        "dropped_invalid": 0,
        "key_alias_used_count": 0,
        "schema_coerce_used_count": 0,
        "symptom_name_normalized_count": 0,
        "item1_gate_kept_count": 0,
        "item1_gate_dropped_count": 0,
        "item1_gate_soft_clamped_count": 0,
        "precision_gate_dropped_count": 0,
        "precision_gate_soft_clamped_count": 0,
        "precision_gate_item_counts": {},
        "threshold_dropped_count": 0,
        "item9_direct_match_count": 0,
        "item9_passive_risk_match_count": 0,
        "item9_routed_risk_recovery_applied_count": 0,
    }

    candidate_items = [dict(item) for item in items if isinstance(item, dict)]
    for raw_item in candidate_items:
        normalized_item, alias_hits = _normalize_item_keys(
            raw_item,
            key_aliases_enabled=key_aliases_enabled,
        )
        stats["key_alias_used_count"] += int(alias_hits)

        normalized_item, schema_hits = _coerce_schema_defaults(
            normalized_item,
            strict_schema_coerce=strict_schema_coerce,
        )
        stats["schema_coerce_used_count"] += int(schema_hits)

        if method_override:
            normalized_item["method"] = method_override

        symptom_name = str(normalized_item.get("symptom_name", "")).strip()
        resolved_item_id = _coerce_item_id(normalized_item.get("item_id"), symptom_name)
        if resolved_item_id is None:
            stats["dropped_unknown"] += 1
            continue

        canonical_symptom_name, normalized_symptom = _canonicalize_symptom_name(
            resolved_item_id,
            symptom_name,
        )
        normalized_item["symptom_name"] = canonical_symptom_name
        if normalized_symptom:
            stats["symptom_name_normalized_count"] += 1
        if "item_id" not in normalized_item:
            normalized_item["item_id"] = resolved_item_id

        if not _number_in_range(normalized_item.get("intensity"), 0.0, 3.0):
            stats["dropped_invalid"] += 1
            continue
        if not _number_in_range(normalized_item.get("confidence"), 0.0, 1.0):
            stats["dropped_invalid"] += 1
            continue

        record = _coerce_evidence_record(node_name, turn, normalized_item, latest_message)
        if record is None:
            stats["dropped_unknown"] += 1
            continue
        if method_override and str(record.method).strip().lower() != str(method_override).strip().lower():
            record = record.model_copy(update={"method": method_override})

        if _is_item1_llm_candidate(record):
            gated_record, gate_action = _apply_item1_gate(
                record,
                latest_message=latest_message,
                strict_gate=item1_strict_gate,
                weak_max_conf=item1_weak_max_conf,
                weak_max_intensity=item1_weak_max_intensity,
            )
            if gate_action == "dropped":
                stats["item1_gate_dropped_count"] += 1
                continue
            if gate_action == "soft_clamped":
                stats["item1_gate_soft_clamped_count"] += 1
            else:
                stats["item1_gate_kept_count"] += 1
            record = gated_record

        if record is not None:
            if int(record.item_id) == 9:
                item9_signal = _item9_precision_signal(f"{record.evidence_text}\n{latest_message}")
                if item9_signal == "direct":
                    stats["item9_direct_match_count"] += 1
                elif item9_signal == "passive_risk":
                    stats["item9_passive_risk_match_count"] += 1
            gated_record, precision_action = _apply_precision_gate(
                record,
                latest_message=latest_message,
                force_guard=force_guard,
                guard_buckets=guard_buckets,
                routed_risk_item9_allowed=(
                    bool(routed_risk_item9_allowed)
                    and int(record.item_id) == 9
                    and str(record.method or "").strip().lower() == "llm_extractor"
                ),
            )
            if precision_action == "dropped":
                stats["precision_gate_dropped_count"] += 1
                _merge_precision_gate_counts(
                    stats["precision_gate_item_counts"],
                    item_id=int(record.item_id),
                    action=precision_action,
                )
                continue
            if precision_action == "soft_clamped":
                stats["precision_gate_soft_clamped_count"] += 1
                _merge_precision_gate_counts(
                    stats["precision_gate_item_counts"],
                    item_id=int(record.item_id),
                    action=precision_action,
                )
            if (
                gated_record is not None
                and int(record.item_id) == 9
                and str(getattr(gated_record, "precision_gate_action", "") or "") == "kept_routed_risk_passive"
            ):
                stats["item9_routed_risk_recovery_applied_count"] += 1
            record = gated_record

        if record is None:
            continue
        if min_confidence is not None and float(record.confidence) < float(min_confidence):
            stats["threshold_dropped_count"] += 1
            continue
        if min_intensity is not None and float(record.intensity) < float(min_intensity):
            stats["threshold_dropped_count"] += 1
            continue
        records.append(record)

    records.sort(key=lambda record: (float(record.confidence), float(record.intensity)), reverse=True)
    if max_records is not None:
        records = records[: max(0, int(max_records))]
    return records, stats


def _records_from_scored_items(
    scored_items: Sequence[Dict[str, Any]],
    *,
    allowed_item_ids: Sequence[int],
    node_name: str,
    turn: int,
    latest_message: str,
    key_aliases_enabled: bool,
    strict_schema_coerce: bool,
    item1_strict_gate: bool,
    item1_weak_max_conf: float,
    item1_weak_max_intensity: float,
    method_override: str = "llm_extractor",
    force_guard: bool = False,
    guard_buckets: set[str] | None = None,
    routed_risk_item9_allowed: bool = False,
    min_confidence: float | None = None,
    min_intensity: float | None = None,
    max_records: int | None = None,
    stats_prefix: str = "detail",
    current_detector_question: str = "",
) -> tuple[List[EvidenceRecord], Dict[str, Any]]:
    def _key(name: str) -> str:
        return f"{stats_prefix}_{name}"

    allowed_order = [int(item_id) for item_id in allowed_item_ids]
    allowed = set(allowed_order)
    stats: Dict[str, Any] = {
        "dropped_unknown": 0,
        "dropped_invalid": 0,
        "key_alias_used_count": 0,
        "schema_coerce_used_count": 0,
        "symptom_name_normalized_count": 0,
        "item1_gate_kept_count": 0,
        "item1_gate_dropped_count": 0,
        "item1_gate_soft_clamped_count": 0,
        "precision_gate_dropped_count": 0,
        "precision_gate_soft_clamped_count": 0,
        "precision_gate_item_counts": {},
        "threshold_dropped_count": 0,
        "out_of_scope_count": 0,
        "item9_direct_match_count": 0,
        "item9_passive_risk_match_count": 0,
        "item9_routed_risk_recovery_applied_count": 0,
        _key("scored_item_count"): 0,
        _key("supported_item_count"): 0,
        _key("unsupported_item_count"): 0,
        _key("supported_item_ids"): [],
        _key("missing_allowed_item_count"): 0,
        _key("supported_rows_dropped_by_item1"): 0,
        _key("supported_rows_dropped_by_item9"): 0,
        _key("supported_rows_kept_post_validation"): 0,
        _key("module3_soft_support_count"): 0,
        _key("module3_soft_support_item_ids"): [],
        _key("item14_worthlessness_hint_applied"): 0,
        _key("item14_latent_support_applied"): 0,
        _key("item21_mild_direct_keep_applied"): 0,
        _key("item21_direct_denial_blocked"): 0,
        _key("item18_change_signal_match"): 0,
        _key("item18_change_signal_rejected"): 0,
    }

    supported_items: List[Dict[str, Any]] = []
    seen_item_ids: set[int] = set()
    scorer_supported_item_ids: set[int] = set()
    module3_soft_support_item_ids = set()
    apply_scoped_item14_hint = False
    apply_scoped_item14_latent_support = False
    item14_identity_change_signal = False
    apply_scoped_item21_keep = False
    item21_direct_denial_blocked = False
    item18_change_signal_match = _item18_has_direct_change_signal(
        latest_message,
        previous_question=current_detector_question,
    )
    if str(method_override or "").strip().lower() == "llm_extractor":
        module3_soft_support_item_ids = set(_module3_soft_support_item_ids(latest_message))
        apply_scoped_item14_hint = 14 in allowed and _has_item14_worthlessness_semantics(latest_message)
        apply_scoped_item14_latent_support = 14 in allowed and _has_item14_latent_support_semantics(latest_message)
        item14_identity_change_signal = 14 in allowed and _has_item14_identity_change_semantics(latest_message)
        item21_direct_denial_blocked = (
            21 in allowed
            and _is_item21_detector_question(current_detector_question)
            and (
                _is_clear_no_symptom_reply(latest_message, previous_question=current_detector_question)
                or _has_item21_direct_denial(latest_message)
            )
        )
        apply_scoped_item21_keep = (
            21 in allowed
            and _is_item21_detector_question(current_detector_question)
            and _has_item21_mild_direct_signal(latest_message)
            and not item21_direct_denial_blocked
        )

    for raw_item in [dict(item) for item in scored_items if isinstance(item, dict)]:
        normalized_item, alias_hits = _normalize_item_keys(
            raw_item,
            key_aliases_enabled=key_aliases_enabled,
        )
        stats["key_alias_used_count"] += int(alias_hits)

        normalized_item, schema_hits = _coerce_scored_schema_defaults(
            normalized_item,
            strict_schema_coerce=strict_schema_coerce,
        )
        stats["schema_coerce_used_count"] += int(schema_hits)

        raw_symptom_name = str(normalized_item.get("symptom_name", "")).strip()
        resolved_item_id = _coerce_item_id(normalized_item.get("item_id"), raw_symptom_name)
        if resolved_item_id is None:
            stats["dropped_unknown"] += 1
            continue
        if int(resolved_item_id) not in allowed:
            stats["out_of_scope_count"] += 1
            continue
        if int(resolved_item_id) in seen_item_ids:
            continue
        seen_item_ids.add(int(resolved_item_id))

        canonical_symptom_name, normalized_symptom = _canonicalize_symptom_name(
            int(resolved_item_id),
            raw_symptom_name,
        )
        if normalized_symptom:
            stats["symptom_name_normalized_count"] += 1

        supported = _coerce_supported_flag(normalized_item.get("supported", False))
        if supported:
            scorer_supported_item_ids.add(int(resolved_item_id))
        module3_soft_supported = False
        item14_worthlessness_hint_applied = False
        item14_latent_support_applied = False
        if (
            not supported
            and int(resolved_item_id) == 14
            and apply_scoped_item14_latent_support
            and not (item14_identity_change_signal and 7 in scorer_supported_item_ids)
        ):
            supported = True
            item14_latent_support_applied = True
            item14_worthlessness_hint_applied = apply_scoped_item14_hint
        if (
            not supported
            and int(resolved_item_id) in PRECISION_GUARD_MODULE3
            and int(resolved_item_id) in module3_soft_support_item_ids
        ):
            supported = True
            module3_soft_supported = True
        item21_mild_direct_keep_applied = False
        if not supported and int(resolved_item_id) == 21 and apply_scoped_item21_keep:
            supported = True
            item21_mild_direct_keep_applied = True
        if (
            not supported
            and int(resolved_item_id) == 21
            and item21_direct_denial_blocked
        ):
            stats[_key("item21_direct_denial_blocked")] = int(
                stats[_key("item21_direct_denial_blocked")]
            ) + 1
        if supported and int(resolved_item_id) == 18 and not item18_change_signal_match:
            supported = False
            stats[_key("item18_change_signal_rejected")] = int(stats[_key("item18_change_signal_rejected")]) + 1
        if not supported:
            continue

        if module3_soft_supported:
            try:
                normalized_intensity = float(normalized_item.get("intensity", 0.0) or 0.0)
            except (TypeError, ValueError):
                normalized_intensity = 0.0
            try:
                normalized_confidence = float(normalized_item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                normalized_confidence = 0.0
            normalized_item["intensity"] = max(normalized_intensity, 1.0)
            normalized_item["confidence"] = max(normalized_confidence, 0.38)
            normalized_item["reason"] = str(normalized_item.get("reason", "") or "").strip() or (
                "module3 soft support from explicit self-evaluation language"
            )
            stats[_key("module3_soft_support_count")] = int(stats[_key("module3_soft_support_count")]) + 1
            soft_ids = list(stats[_key("module3_soft_support_item_ids")])
            if int(resolved_item_id) not in soft_ids:
                soft_ids.append(int(resolved_item_id))
            stats[_key("module3_soft_support_item_ids")] = sorted(soft_ids)
        if item14_latent_support_applied:
            normalized_item["intensity"] = max(float(normalized_item.get("intensity", 0.0) or 0.0), 1.0)
            normalized_item["confidence"] = max(float(normalized_item.get("confidence", 0.0) or 0.0), 0.45)
            normalized_item["reason"] = str(normalized_item.get("reason", "") or "").strip() or (
                "item14 latent worth/identity support from explicit self-evaluation language"
            )
            stats[_key("item14_latent_support_applied")] = int(
                stats[_key("item14_latent_support_applied")]
            ) + 1
        if item14_worthlessness_hint_applied:
            stats[_key("item14_worthlessness_hint_applied")] = int(
                stats[_key("item14_worthlessness_hint_applied")]
            ) + 1
        if item21_mild_direct_keep_applied:
            normalized_item["intensity"] = max(float(normalized_item.get("intensity", 0.0) or 0.0), 1.0)
            normalized_item["confidence"] = max(float(normalized_item.get("confidence", 0.0) or 0.0), 0.40)
            normalized_item["reason"] = str(normalized_item.get("reason", "") or "").strip() or (
                "item21 mild direct keep from explicit sexual-interest decrease phrasing"
            )
            stats[_key("item21_mild_direct_keep_applied")] = int(
                stats[_key("item21_mild_direct_keep_applied")]
            ) + 1
        if int(resolved_item_id) == 18 and item18_change_signal_match:
            stats[_key("item18_change_signal_match")] = int(stats[_key("item18_change_signal_match")]) + 1

        supported_items.append(
            {
                "item_id": int(resolved_item_id),
                "symptom_name": canonical_symptom_name,
                "direction": "increase",
                "intensity": normalized_item.get("intensity", 0.0),
                "confidence": normalized_item.get("confidence", 0.0),
                "evidence_text": str(normalized_item.get("anchor_quote", "")).strip() or latest_message[:220],
                "reason": str(normalized_item.get("reason", "")).strip() or f"supported {stats_prefix} scorer item",
                "method": method_override,
            }
        )

    stats[_key("scored_item_count")] = len(seen_item_ids)
    supported_item_ids = [int(item["item_id"]) for item in supported_items]
    stats[_key("supported_item_count")] = len(supported_item_ids)
    stats[_key("unsupported_item_count")] = int(stats[_key("scored_item_count")]) - int(
        stats[_key("supported_item_count")]
    )
    stats[_key("supported_item_ids")] = sorted(supported_item_ids)
    stats[_key("missing_allowed_item_count")] = max(0, len(allowed_order) - len(seen_item_ids))

    records, row_stats = _records_from_llm_items(
        supported_items,
        node_name=node_name,
        turn=turn,
        latest_message=latest_message,
        key_aliases_enabled=key_aliases_enabled,
        strict_schema_coerce=False,
        item1_strict_gate=item1_strict_gate,
        item1_weak_max_conf=item1_weak_max_conf,
        item1_weak_max_intensity=item1_weak_max_intensity,
        method_override=method_override,
        force_guard=force_guard,
        guard_buckets=guard_buckets,
        routed_risk_item9_allowed=routed_risk_item9_allowed,
        min_confidence=min_confidence,
        min_intensity=min_intensity,
        max_records=max_records,
    )
    for key in (
        "dropped_unknown",
        "dropped_invalid",
        "key_alias_used_count",
        "schema_coerce_used_count",
        "symptom_name_normalized_count",
        "item1_gate_kept_count",
        "item1_gate_dropped_count",
        "item1_gate_soft_clamped_count",
        "precision_gate_dropped_count",
        "precision_gate_soft_clamped_count",
        "threshold_dropped_count",
        "item9_direct_match_count",
        "item9_passive_risk_match_count",
        "item9_routed_risk_recovery_applied_count",
    ):
        stats[key] += int(row_stats.get(key, 0) or 0)
    _merge_precision_gate_item_counts(stats["precision_gate_item_counts"], row_stats["precision_gate_item_counts"])
    stats[_key("supported_rows_dropped_by_item1")] = int(row_stats["item1_gate_dropped_count"] or 0)
    stats[_key("supported_rows_dropped_by_item9")] = int(
        row_stats["precision_gate_item_counts"].get("9", {}).get("dropped", 0) or 0
    )
    stats[_key("supported_rows_kept_post_validation")] = len(records)
    return records, stats


def _likelihood_from_record(record: EvidenceRecord) -> List[float]:
    strength = max(0.0, min(1.0, float(record.confidence) * max(0.1, float(record.intensity) / 3.0)))

    if record.direction == "increase":
        return [
            max(0.05, 1.0 - (0.8 * strength)),
            max(0.10, 1.0 - (0.4 * strength)),
            1.0 + (0.4 * strength),
            1.0 + (0.8 * strength),
        ]
    if record.direction == "decrease":
        return [
            1.0 + (0.8 * strength),
            1.0 + (0.4 * strength),
            max(0.10, 1.0 - (0.4 * strength)),
            max(0.05, 1.0 - (0.8 * strength)),
        ]

    neutral = 1.0 + (0.15 * strength)
    return [neutral, neutral, neutral, neutral]


def _extract_likelihoods_v2(
    state: AgentState,
    *,
    turn: int,
    latest_message: str,
    node_name: str,
) -> Dict[str, Any]:
    target_spec = _extract_target_spec(state, node_name)
    allowed_item_ids = list(target_spec["allowed_item_ids"])
    evidence_records: List[EvidenceRecord] = []
    counters = dict(state.get("failure_counters", {}))

    source = "llm_extractor"
    raw_nonempty = False
    json_parse_ok = False
    raw_items_count = 0
    dropped_unknown = 0
    dropped_invalid = 0
    lexical_prefilter: List[EvidenceRecord] = []
    salvage_used = False
    salvage_items_count = 0
    key_alias_used_count = 0
    schema_coerce_used_count = 0
    symptom_name_normalized_count = 0
    item1_gate_kept_count = 0
    item1_gate_dropped_count = 0
    item1_gate_soft_clamped_count = 0
    precision_gate_dropped_count = 0
    precision_gate_soft_clamped_count = 0
    precision_gate_item_counts: Dict[str, Dict[str, int]] = {}
    parse_error_kind = ""
    parse_error_message = ""
    parse_error_line = 0
    parse_error_column = 0
    parse_error_position = 0
    parse_balance: Dict[str, Any] = {}
    llm_called = False
    raw_payload_logged = ""
    gate_called = False
    gate_parse_ok = False
    gate_target_relevant = False
    gate_candidate_item_ids: List[int] = []
    gate_confidence = 0.0
    stage2_called = False
    stage2_parse_ok = False
    out_of_scope_item_count = 0
    parse_fail_stage = ""
    clear_no_symptom_skip = False
    gate_soft_false_overridden = False
    detail_called_due_to_gate_false = False
    detail_called_due_to_gate_parse_fail = False
    detail_empty_after_gate_true = False
    detail_empty_after_gate_false = False
    detail_scored_item_count = 0
    detail_supported_item_count = 0
    detail_unsupported_item_count = 0
    detail_supported_item_ids: List[int] = []
    detail_missing_allowed_item_count = 0
    detail_supported_rows_dropped_by_item1 = 0
    detail_supported_rows_dropped_by_item9 = 0
    detail_supported_rows_kept_post_validation = 0
    detail_module3_soft_support_count = 0
    detail_module3_soft_support_item_ids: List[int] = []
    detail_item14_worthlessness_hint_applied = False
    detail_item14_latent_support_applied = False
    detail_item21_mild_direct_keep_applied = False
    detail_item21_direct_denial_blocked = False
    detail_item18_change_signal_match = False
    detail_item18_change_signal_rejected = False
    item9_direct_match = False
    item9_passive_risk_match = False
    item9_routed_risk_recovery_applied = False
    genuine_no_signal_turn = False
    opportunistic_called = False
    opportunistic_parse_ok = False
    opportunistic_skipped_on_risk = False
    opportunistic_shortlist_called = False
    opportunistic_shortlist_parse_ok = False
    opportunistic_has_strong_offtarget_signal = False
    opportunistic_candidate_item_ids: List[int] = []
    opportunistic_score_called = False
    opportunistic_score_parse_ok = False
    opportunistic_raw_items_count = 0
    opportunistic_kept_items_count = 0
    opportunistic_dropped_weak_count = 0
    opportunistic_salvage_used = False
    opportunistic_item_ids: List[int] = []
    opportunistic_scored_item_count = 0
    opportunistic_supported_item_count = 0
    opportunistic_unsupported_item_count = 0
    opportunistic_supported_item_ids: List[int] = []
    opportunistic_missing_item_count = 0
    llm_on_lexical_hit = _env_bool("EVIDENCE_LLM_ON_LEXICAL_HIT", "0")
    key_aliases_enabled = _env_bool("EXTRACTOR_JSON_KEY_ALIASES", "1")
    strict_schema_coerce = _env_bool("EXTRACTOR_STRICT_SCHEMA_COERCE", "1")
    item1_strict_gate = _env_bool("EXTRACT_ITEM1_STRICT_GATE", "0")
    item1_weak_max_conf = _clamp(_env_float("EXTRACT_ITEM1_WEAK_MAX_CONF", 0.55), 0.0, 1.0)
    item1_weak_max_intensity = _clamp(_env_float("EXTRACT_ITEM1_WEAK_MAX_INTENSITY", 1.5), 0.0, 3.0)
    extractor_min_records_target = max(1, int(os.getenv("EXTRACTOR_MIN_RECORDS_TARGET", "1")))
    llm_raw_text = ""
    previous_question = _previous_detector_question(state)
    genuine_no_signal_turn = _is_clear_no_symptom_reply(latest_message, previous_question=previous_question)

    if latest_message.strip():
        lexical_prefilter = _fallback_evidence_from_text(node_name, turn, latest_message)
        lexical_prefilter, dropped_count, soft_clamped_count, item_counts = _apply_precision_gate_batch(
            lexical_prefilter,
            latest_message=latest_message,
        )
        precision_gate_dropped_count += int(dropped_count)
        precision_gate_soft_clamped_count += int(soft_clamped_count)
        _merge_precision_gate_item_counts(precision_gate_item_counts, item_counts)
        lexical_prefilter, lexical_scope_dropped = _filter_records_to_allowed_items(
            lexical_prefilter,
            allowed_item_ids=allowed_item_ids,
        )
        out_of_scope_item_count += int(lexical_scope_dropped)

        should_skip_llm = len(lexical_prefilter) >= extractor_min_records_target and not llm_on_lexical_hit
        if should_skip_llm:
            evidence_records = lexical_prefilter
            source = "lexical_prefilter"
        elif genuine_no_signal_turn:
            clear_no_symptom_skip = True
            source = "clear_no_symptom_skip"
        else:
            gate_called = True
            llm_called = True
            gate_prompt = get_prompt("evidence_gate").format(
                node_name=target_spec["route"],
                target_item_id=target_spec["target_item_id"],
                target_item_name=target_spec["target_item_name"],
                target_module_id=target_spec["target_module_id"],
                target_module_name=target_spec["target_module_name"],
                allowed_item_ids=allowed_item_ids,
                current_detector_question=previous_question or "none",
                recent_context=_recent_context(state) or "none",
                latest_message=latest_message,
            )
            try:
                llm = get_extractor_llm()
                gate_raw = str(llm.invoke([("system", gate_prompt)]).content or "")
                raw_nonempty = bool(gate_raw.strip())
                if not raw_nonempty:
                    counters = bump_failure_counter(counters, "extract_llm_empty_payload")
                gate_parsed, gate_parse_ok, gate_diagnostics = _parse_json_payload(gate_raw)
                gate_payload = _coerce_gate_payload(gate_parsed, allowed_item_ids=allowed_item_ids)
                gate_target_relevant = bool(gate_payload["target_relevant"])
                gate_candidate_item_ids = list(gate_payload["candidate_item_ids"])
                gate_confidence = float(gate_payload["confidence"])
                parse_balance = {
                    "brace_open": int(gate_diagnostics.get("brace_open", 0) or 0),
                    "brace_close": int(gate_diagnostics.get("brace_close", 0) or 0),
                    "bracket_open": int(gate_diagnostics.get("bracket_open", 0) or 0),
                    "bracket_close": int(gate_diagnostics.get("bracket_close", 0) or 0),
                    "double_quote_count": int(gate_diagnostics.get("double_quote_count", 0) or 0),
                    "unmatched_double_quote": bool(gate_diagnostics.get("unmatched_double_quote", False)),
                }
                if not gate_parse_ok:
                    detail_called_due_to_gate_parse_fail = True
                elif not gate_target_relevant:
                    gate_soft_false_overridden = True
                    detail_called_due_to_gate_false = True

                stage2_called = True
                if stage2_called:
                    detail_prompt = get_prompt("evidence_extract_targeted").format(
                        node_name=target_spec["route"],
                        target_item_id=target_spec["target_item_id"],
                        target_item_name=target_spec["target_item_name"],
                        target_module_id=target_spec["target_module_id"],
                        target_module_name=target_spec["target_module_name"],
                        allowed_item_ids=allowed_item_ids,
                        candidate_item_ids=gate_candidate_item_ids,
                        anchor_quote=gate_payload["anchor_quote"] or "none",
                        current_detector_question=previous_question or "none",
                        recent_context=_recent_context(state) or "none",
                        latest_message=latest_message,
                    )
                    detail_raw = str(llm.invoke([("system", detail_prompt)]).content or "")
                    llm_raw_text = detail_raw
                    raw_nonempty = bool(detail_raw.strip())
                    if not raw_nonempty:
                        counters = bump_failure_counter(counters, "extract_llm_empty_payload")
                        source = "llm_detail_empty_payload"
                    parsed, stage2_parse_ok, parse_diagnostics = _parse_json_payload(detail_raw)
                    json_parse_ok = stage2_parse_ok
                    parse_balance = {
                        "brace_open": int(parse_diagnostics.get("brace_open", 0) or 0),
                        "brace_close": int(parse_diagnostics.get("brace_close", 0) or 0),
                        "bracket_open": int(parse_diagnostics.get("bracket_open", 0) or 0),
                        "bracket_close": int(parse_diagnostics.get("bracket_close", 0) or 0),
                        "double_quote_count": int(parse_diagnostics.get("double_quote_count", 0) or 0),
                        "unmatched_double_quote": bool(parse_diagnostics.get("unmatched_double_quote", False)),
                    }
                    parse_error_kind = str(parse_diagnostics.get("error_kind", "") or "")
                    parse_error_message = str(parse_diagnostics.get("error_message", "") or "")
                    parse_error_line = int(parse_diagnostics.get("error_line", 0) or 0)
                    parse_error_column = int(parse_diagnostics.get("error_column", 0) or 0)
                    parse_error_position = int(parse_diagnostics.get("error_position", 0) or 0)

                    scored_items, schema_payload_coerce = _payload_scored_items(parsed)
                    schema_coerce_used_count += int(schema_payload_coerce)
                    if schema_payload_coerce > 0:
                        counters = bump_failure_counter(
                            counters,
                            "extract_schema_coerce_used",
                            amount=schema_payload_coerce,
                        )

                    if raw_nonempty and not scored_items:
                        salvage_items = _salvage_items_from_text(detail_raw)
                        salvage_items, salvage_scope_dropped = _filter_items_to_allowed_scope(
                            salvage_items,
                            allowed_item_ids=allowed_item_ids,
                        )
                        out_of_scope_item_count += int(salvage_scope_dropped)
                        if salvage_items:
                            items = salvage_items
                            salvage_used = True
                            salvage_items_count = len(salvage_items)
                            source = "llm_salvage"
                            counters = bump_failure_counter(counters, "extract_salvage_used")
                            counters = bump_failure_counter(
                                counters, "extract_salvage_kept_items", amount=salvage_items_count
                            )

                    raw_items_count = len(scored_items) if isinstance(scored_items, list) else 0
                    if isinstance(scored_items, list):
                        if salvage_used:
                            evidence_records, salvage_stats = _records_from_llm_items(
                                salvage_items,
                                node_name=node_name,
                                turn=turn,
                                latest_message=latest_message,
                                key_aliases_enabled=key_aliases_enabled,
                                strict_schema_coerce=strict_schema_coerce,
                                item1_strict_gate=item1_strict_gate,
                                item1_weak_max_conf=item1_weak_max_conf,
                                item1_weak_max_intensity=item1_weak_max_intensity,
                                force_guard=True,
                            )
                            dropped_unknown += int(salvage_stats["dropped_unknown"])
                            dropped_invalid += int(salvage_stats["dropped_invalid"])
                            key_alias_used_count += int(salvage_stats["key_alias_used_count"])
                            schema_coerce_used_count += int(salvage_stats["schema_coerce_used_count"])
                            symptom_name_normalized_count += int(salvage_stats["symptom_name_normalized_count"])
                            item1_gate_kept_count += int(salvage_stats["item1_gate_kept_count"])
                            item1_gate_dropped_count += int(salvage_stats["item1_gate_dropped_count"])
                            item1_gate_soft_clamped_count += int(salvage_stats["item1_gate_soft_clamped_count"])
                            precision_gate_dropped_count += int(salvage_stats["precision_gate_dropped_count"])
                            precision_gate_soft_clamped_count += int(
                                salvage_stats["precision_gate_soft_clamped_count"]
                            )
                            _merge_precision_gate_item_counts(
                                precision_gate_item_counts,
                                salvage_stats["precision_gate_item_counts"],
                            )
                        else:
                            evidence_records, scored_stats = _records_from_scored_items(
                                scored_items,
                                allowed_item_ids=allowed_item_ids,
                                node_name=node_name,
                                turn=turn,
                                latest_message=latest_message,
                                key_aliases_enabled=key_aliases_enabled,
                                strict_schema_coerce=strict_schema_coerce,
                                item1_strict_gate=item1_strict_gate,
                                item1_weak_max_conf=item1_weak_max_conf,
                                item1_weak_max_intensity=item1_weak_max_intensity,
                                method_override="llm_extractor",
                                guard_buckets={"item9"},
                                routed_risk_item9_allowed=(str(node_name).strip().lower() == "risk"),
                                stats_prefix="detail",
                                current_detector_question=previous_question or "",
                            )
                            dropped_unknown += int(scored_stats["dropped_unknown"])
                            dropped_invalid += int(scored_stats["dropped_invalid"])
                            key_alias_used_count += int(scored_stats["key_alias_used_count"])
                            schema_coerce_used_count += int(scored_stats["schema_coerce_used_count"])
                            symptom_name_normalized_count += int(scored_stats["symptom_name_normalized_count"])
                            item1_gate_kept_count += int(scored_stats["item1_gate_kept_count"])
                            item1_gate_dropped_count += int(scored_stats["item1_gate_dropped_count"])
                            item1_gate_soft_clamped_count += int(scored_stats["item1_gate_soft_clamped_count"])
                            precision_gate_dropped_count += int(scored_stats["precision_gate_dropped_count"])
                            precision_gate_soft_clamped_count += int(
                                scored_stats["precision_gate_soft_clamped_count"]
                            )
                            out_of_scope_item_count += int(scored_stats["out_of_scope_count"])
                            detail_scored_item_count = int(scored_stats["detail_scored_item_count"])
                            detail_supported_item_count = int(scored_stats["detail_supported_item_count"])
                            detail_unsupported_item_count = int(scored_stats["detail_unsupported_item_count"])
                            detail_supported_item_ids = list(scored_stats["detail_supported_item_ids"])
                            detail_missing_allowed_item_count = int(
                                scored_stats["detail_missing_allowed_item_count"]
                            )
                            detail_supported_rows_dropped_by_item1 = int(
                                scored_stats["detail_supported_rows_dropped_by_item1"]
                            )
                            detail_supported_rows_dropped_by_item9 = int(
                                scored_stats["detail_supported_rows_dropped_by_item9"]
                            )
                            detail_supported_rows_kept_post_validation = int(
                                scored_stats["detail_supported_rows_kept_post_validation"]
                            )
                            detail_module3_soft_support_count = int(
                                scored_stats["detail_module3_soft_support_count"]
                            )
                            detail_module3_soft_support_item_ids = list(
                                scored_stats["detail_module3_soft_support_item_ids"]
                            )
                            detail_item14_worthlessness_hint_applied = bool(
                                int(scored_stats["detail_item14_worthlessness_hint_applied"] or 0) > 0
                            )
                            detail_item14_latent_support_applied = bool(
                                int(scored_stats["detail_item14_latent_support_applied"] or 0) > 0
                            )
                            detail_item21_mild_direct_keep_applied = bool(
                                int(scored_stats["detail_item21_mild_direct_keep_applied"] or 0) > 0
                            )
                            detail_item21_direct_denial_blocked = bool(
                                int(scored_stats["detail_item21_direct_denial_blocked"] or 0) > 0
                            )
                            detail_item18_change_signal_match = bool(
                                int(scored_stats["detail_item18_change_signal_match"] or 0) > 0
                            )
                            detail_item18_change_signal_rejected = bool(
                                int(scored_stats["detail_item18_change_signal_rejected"] or 0) > 0
                            )
                            item9_direct_match = bool(scored_stats["item9_direct_match_count"] > 0)
                            item9_passive_risk_match = bool(scored_stats["item9_passive_risk_match_count"] > 0)
                            item9_routed_risk_recovery_applied = bool(
                                scored_stats["item9_routed_risk_recovery_applied_count"] > 0
                            )
                            _merge_precision_gate_item_counts(
                                precision_gate_item_counts,
                                scored_stats["precision_gate_item_counts"],
                            )
                    else:
                        source = "llm_extractor_non_list_payload"

                    if not stage2_parse_ok:
                        parse_fail_stage = "detail"
                        source = "llm_detail_parse_fail"
                        raw_payload_logged = detail_raw
                    elif not evidence_records:
                        if gate_parse_ok and gate_target_relevant and detail_supported_item_count == 0:
                            detail_empty_after_gate_true = True
                        elif gate_parse_ok and not gate_target_relevant and detail_supported_item_count == 0:
                            detail_empty_after_gate_false = True
                    elif not salvage_used:
                        source = "llm_extractor"
            except LLMBudgetExceeded:
                raise
            except Exception as exc:
                source = "llm_extractor_error"
                counters = bump_failure_counter(counters, "extract_llm_call_fail")
                error_text = str(exc).strip()
                parse_error_kind = "llm_timeout" if "timed out" in error_text.lower() else "llm_call_failed"
                parse_error_message = error_text[:300]
                parse_fail_stage = "detail" if stage2_called else ("gate" if gate_called else "")
    else:
        source = "skip_empty_message"

    if dropped_unknown > 0:
        counters = bump_failure_counter(counters, "extract_item_map_fail", amount=dropped_unknown)

    fallback_records: List[EvidenceRecord] = []
    if not evidence_records and latest_message.strip() and not clear_no_symptom_skip:
        fallback_records = _fallback_evidence_from_text(node_name, turn, latest_message)
        fallback_records, dropped_count, soft_clamped_count, item_counts = _apply_precision_gate_batch(
            fallback_records,
            latest_message=latest_message,
        )
        precision_gate_dropped_count += int(dropped_count)
        precision_gate_soft_clamped_count += int(soft_clamped_count)
        _merge_precision_gate_item_counts(precision_gate_item_counts, item_counts)
        fallback_records, fallback_scope_dropped = _filter_records_to_allowed_items(
            fallback_records,
            allowed_item_ids=allowed_item_ids,
        )
        out_of_scope_item_count += int(fallback_scope_dropped)
        if fallback_records:
            evidence_records = fallback_records
            source = "lexical_fallback"

    if not evidence_records and latest_message.strip() and not clear_no_symptom_skip:
        if target_spec["route"] == "risk":
            opportunistic_skipped_on_risk = True
        else:
            opportunistic_called = True
            llm_called = True
            opportunistic_shortlist_called = True
            shortlist_prompt = get_prompt("evidence_shortlist_opportunistic").format(
                node_name=target_spec["route"],
                target_item_id=target_spec["target_item_id"],
                target_item_name=target_spec["target_item_name"],
                target_module_id=target_spec["target_module_id"],
                target_module_name=target_spec["target_module_name"],
                scoped_allowed_item_ids=allowed_item_ids,
                current_detector_question=previous_question or "none",
                recent_context=_recent_context(state) or "none",
                latest_message=latest_message,
            )
            try:
                opportunistic_llm = get_extractor_llm()
                shortlist_raw = str(opportunistic_llm.invoke([("system", shortlist_prompt)]).content or "")
                raw_nonempty = bool(shortlist_raw.strip())
                if not raw_nonempty:
                    counters = bump_failure_counter(counters, "extract_llm_empty_payload")
                    source = "llm_opportunistic_empty_payload"
                shortlist_parsed, opportunistic_shortlist_parse_ok, shortlist_diagnostics = _parse_json_payload(
                    shortlist_raw
                )
                parse_balance = {
                    "brace_open": int(shortlist_diagnostics.get("brace_open", 0) or 0),
                    "brace_close": int(shortlist_diagnostics.get("brace_close", 0) or 0),
                    "bracket_open": int(shortlist_diagnostics.get("bracket_open", 0) or 0),
                    "bracket_close": int(shortlist_diagnostics.get("bracket_close", 0) or 0),
                    "double_quote_count": int(shortlist_diagnostics.get("double_quote_count", 0) or 0),
                    "unmatched_double_quote": bool(shortlist_diagnostics.get("unmatched_double_quote", False)),
                }
                parse_error_kind = str(shortlist_diagnostics.get("error_kind", "") or "")
                parse_error_message = str(shortlist_diagnostics.get("error_message", "") or "")
                parse_error_line = int(shortlist_diagnostics.get("error_line", 0) or 0)
                parse_error_column = int(shortlist_diagnostics.get("error_column", 0) or 0)
                parse_error_position = int(shortlist_diagnostics.get("error_position", 0) or 0)

                shortlist_payload = _coerce_opportunistic_shortlist_payload(
                    shortlist_parsed,
                    scoped_allowed_item_ids=allowed_item_ids,
                )
                opportunistic_has_strong_offtarget_signal = bool(
                    shortlist_payload["has_strong_offtarget_signal"]
                )
                opportunistic_candidate_item_ids = list(shortlist_payload["candidate_item_ids"])

                if not opportunistic_shortlist_parse_ok:
                    opportunistic_parse_ok = False
                    parse_fail_stage = "opportunistic_shortlist"
                    source = "llm_opportunistic_parse_fail"
                    raw_payload_logged = shortlist_raw
                elif opportunistic_has_strong_offtarget_signal and opportunistic_candidate_item_ids:
                    opportunistic_score_called = True
                    score_prompt = get_prompt("evidence_score_opportunistic").format(
                        node_name=target_spec["route"],
                        target_item_id=target_spec["target_item_id"],
                        target_item_name=target_spec["target_item_name"],
                        target_module_id=target_spec["target_module_id"],
                        target_module_name=target_spec["target_module_name"],
                        scoped_allowed_item_ids=allowed_item_ids,
                        candidate_item_ids=opportunistic_candidate_item_ids,
                        anchor_quote=shortlist_payload["anchor_quote"] or "none",
                        shortlist_reason=shortlist_payload["reason"] or "none",
                        current_detector_question=previous_question or "none",
                        recent_context=_recent_context(state) or "none",
                        latest_message=latest_message,
                    )
                    opportunistic_raw = str(opportunistic_llm.invoke([("system", score_prompt)]).content or "")
                    raw_nonempty = bool(opportunistic_raw.strip())
                    if not raw_nonempty:
                        counters = bump_failure_counter(counters, "extract_llm_empty_payload")
                        source = "llm_opportunistic_empty_payload"
                    opportunistic_parsed, opportunistic_score_parse_ok, opportunistic_diagnostics = _parse_json_payload(
                        opportunistic_raw
                    )
                    opportunistic_parse_ok = opportunistic_score_parse_ok
                    parse_balance = {
                        "brace_open": int(opportunistic_diagnostics.get("brace_open", 0) or 0),
                        "brace_close": int(opportunistic_diagnostics.get("brace_close", 0) or 0),
                        "bracket_open": int(opportunistic_diagnostics.get("bracket_open", 0) or 0),
                        "bracket_close": int(opportunistic_diagnostics.get("bracket_close", 0) or 0),
                        "double_quote_count": int(opportunistic_diagnostics.get("double_quote_count", 0) or 0),
                        "unmatched_double_quote": bool(opportunistic_diagnostics.get("unmatched_double_quote", False)),
                    }
                    parse_error_kind = str(opportunistic_diagnostics.get("error_kind", "") or "")
                    parse_error_message = str(opportunistic_diagnostics.get("error_message", "") or "")
                    parse_error_line = int(opportunistic_diagnostics.get("error_line", 0) or 0)
                    parse_error_column = int(opportunistic_diagnostics.get("error_column", 0) or 0)
                    parse_error_position = int(opportunistic_diagnostics.get("error_position", 0) or 0)

                    opportunistic_items, opportunistic_schema_coerce = _payload_scored_items(opportunistic_parsed)
                    schema_coerce_used_count += int(opportunistic_schema_coerce)
                    if opportunistic_schema_coerce > 0:
                        counters = bump_failure_counter(
                            counters, "extract_schema_coerce_used", amount=opportunistic_schema_coerce
                        )

                    opportunistic_raw_items_count = (
                        len(opportunistic_items) if isinstance(opportunistic_items, list) else 0
                    )
                    if isinstance(opportunistic_items, list):
                        opportunistic_records, opportunistic_stats = _records_from_scored_items(
                            opportunistic_items,
                            allowed_item_ids=opportunistic_candidate_item_ids,
                            node_name=node_name,
                            turn=turn,
                            latest_message=latest_message,
                            key_aliases_enabled=key_aliases_enabled,
                            strict_schema_coerce=strict_schema_coerce,
                            item1_strict_gate=item1_strict_gate,
                            item1_weak_max_conf=item1_weak_max_conf,
                            item1_weak_max_intensity=item1_weak_max_intensity,
                            method_override="llm_opportunistic",
                            force_guard=True,
                            min_confidence=0.55,
                            min_intensity=1.5,
                            max_records=2,
                            stats_prefix="opportunistic",
                            current_detector_question=previous_question or "",
                        )
                        dropped_unknown += int(opportunistic_stats["dropped_unknown"])
                        dropped_invalid += int(opportunistic_stats["dropped_invalid"])
                        key_alias_used_count += int(opportunistic_stats["key_alias_used_count"])
                        schema_coerce_used_count += int(opportunistic_stats["schema_coerce_used_count"])
                        symptom_name_normalized_count += int(opportunistic_stats["symptom_name_normalized_count"])
                        item1_gate_kept_count += int(opportunistic_stats["item1_gate_kept_count"])
                        item1_gate_dropped_count += int(opportunistic_stats["item1_gate_dropped_count"])
                        item1_gate_soft_clamped_count += int(opportunistic_stats["item1_gate_soft_clamped_count"])
                        precision_gate_dropped_count += int(opportunistic_stats["precision_gate_dropped_count"])
                        precision_gate_soft_clamped_count += int(
                            opportunistic_stats["precision_gate_soft_clamped_count"]
                        )
                        opportunistic_dropped_weak_count += int(opportunistic_stats["threshold_dropped_count"])
                        opportunistic_scored_item_count += int(opportunistic_stats["opportunistic_scored_item_count"])
                        opportunistic_supported_item_count += int(
                            opportunistic_stats["opportunistic_supported_item_count"]
                        )
                        opportunistic_unsupported_item_count += int(
                            opportunistic_stats["opportunistic_unsupported_item_count"]
                        )
                        opportunistic_supported_item_ids = list(
                            opportunistic_stats["opportunistic_supported_item_ids"]
                        )
                        opportunistic_missing_item_count += int(
                            opportunistic_stats["opportunistic_missing_allowed_item_count"]
                        )
                        out_of_scope_item_count += int(opportunistic_stats["out_of_scope_count"])
                        _merge_precision_gate_item_counts(
                            precision_gate_item_counts,
                            opportunistic_stats["precision_gate_item_counts"],
                        )
                        if int(opportunistic_stats["key_alias_used_count"]) > 0:
                            counters = bump_failure_counter(
                                counters,
                                "extract_key_alias_used",
                                amount=int(opportunistic_stats["key_alias_used_count"]),
                            )
                        if int(opportunistic_stats["schema_coerce_used_count"]) > 0:
                            counters = bump_failure_counter(
                                counters,
                                "extract_schema_coerce_used",
                                amount=int(opportunistic_stats["schema_coerce_used_count"]),
                            )
                        if int(opportunistic_stats["symptom_name_normalized_count"]) > 0:
                            counters = bump_failure_counter(
                                counters,
                                "extract_symptom_name_normalized",
                                amount=int(opportunistic_stats["symptom_name_normalized_count"]),
                            )
                        if opportunistic_records:
                            evidence_records = opportunistic_records
                            opportunistic_kept_items_count = len(opportunistic_records)
                            opportunistic_item_ids = [int(record.item_id) for record in opportunistic_records]
                            source = "llm_opportunistic"
                        else:
                            source = "llm_opportunistic"
                    else:
                        source = "llm_opportunistic_non_list_payload"

                    if not opportunistic_score_parse_ok:
                        parse_fail_stage = "opportunistic_score"
                        source = "llm_opportunistic_parse_fail"
                        raw_payload_logged = opportunistic_raw
                    elif not evidence_records:
                        raw_payload_logged = opportunistic_raw
                else:
                    opportunistic_parse_ok = True
                    source = "llm_opportunistic"
                    raw_payload_logged = shortlist_raw
            except LLMBudgetExceeded:
                raise
            except Exception as exc:
                source = "llm_opportunistic_error"
                counters = bump_failure_counter(counters, "extract_llm_call_fail")
                error_text = str(exc).strip()
                parse_error_kind = "llm_timeout" if "timed out" in error_text.lower() else "llm_call_failed"
                parse_error_message = error_text[:300]
                parse_fail_stage = "opportunistic_score" if opportunistic_score_called else "opportunistic_shortlist"

    if gate_called and not stage2_called:
        json_parse_ok = gate_parse_ok
    elif not llm_called:
        json_parse_ok = False

    if llm_called and raw_nonempty and not evidence_records and not raw_payload_logged:
        raw_payload_logged = llm_raw_text
    if parse_fail_stage and raw_nonempty and not evidence_records:
        counters = bump_failure_counter(counters, "extract_json_parse_fail")

    if not evidence_records:
        counters = bump_failure_counter(counters, "extract_empty")
        empty_streak = int(state.get("empty_evidence_streak", 0)) + 1
    else:
        empty_streak = 0

    item14_cross_scope_injected = False
    existing_item_ids = {int(r.item_id) for r in evidence_records}
    if (
        14 not in existing_item_ids
        and 14 not in allowed_item_ids
        and latest_message.strip()
        and _has_item14_worthlessness_semantics(latest_message)
    ):
        for pattern in ITEM14_WORTHLESSNESS_PATTERNS:
            match = pattern.search(latest_message.lower())
            if match:
                evidence_records.append(
                    EvidenceRecord(
                        turn=turn,
                        node=node_name if node_name in {"somatic", "cognitive", "risk"} else "cognitive",
                        item_id=14,
                        symptom_name=BDI_ITEM_NAMES.get(14, "Worthlessness"),
                        direction="increase",
                        intensity=1.50,
                        confidence=0.55,
                        evidence_text=match.group(0),
                        reason=f"item14 cross-scope worthlessness injection: {match.group(0)}",
                        method="llm_opportunistic",
                    )
                )
                item14_cross_scope_injected = True
                break

    likelihood_rows: List[LikelihoodEvidence] = []
    for record in evidence_records:
        method = str(record.method or "llm_extractor")
        likelihood_rows.append(
            LikelihoodEvidence(
                item_id=int(record.item_id),
                likelihood=_likelihood_from_record(record),
                spans=[record.evidence_text],
                extract_confidence=float(record.confidence),
                extract_intensity=float(record.intensity),
                evidence_type=method,
                symptom_name=str(record.symptom_name),
                direction=str(record.direction),
                evidence_id=_evidence_id(record),
                method_weight_hint=float(METHOD_WEIGHT_HINTS.get(method, 0.50)),
                precision_gate_action=str(getattr(record, "precision_gate_action", "kept") or "kept"),
                support_increment_blocked=bool(getattr(record, "support_increment_blocked", False)),
            )
        )

    trace_payload = {
        "turn": turn,
        "source": source,
        "extractor_version": "v2",
        "target_item_id": int(target_spec["target_item_id"]),
        "target_module_id": int(target_spec["target_module_id"]),
        "allowed_item_ids": allowed_item_ids,
        "gate_called": gate_called,
        "gate_parse_ok": gate_parse_ok,
        "gate_target_relevant": gate_target_relevant,
        "gate_candidate_item_ids": gate_candidate_item_ids,
        "gate_confidence": round(float(gate_confidence), 4),
        "stage2_called": stage2_called,
        "stage2_parse_ok": stage2_parse_ok,
        "out_of_scope_item_count": int(out_of_scope_item_count),
        "parse_fail_stage": parse_fail_stage,
        "clear_no_symptom_skip": clear_no_symptom_skip,
        "genuine_no_signal_turn": genuine_no_signal_turn,
        "gate_soft_false_overridden": gate_soft_false_overridden,
        "detail_called_due_to_gate_false": detail_called_due_to_gate_false,
        "detail_called_due_to_gate_parse_fail": detail_called_due_to_gate_parse_fail,
        "detail_empty_after_gate_true": detail_empty_after_gate_true,
        "detail_empty_after_gate_false": detail_empty_after_gate_false,
        "detail_scored_item_count": detail_scored_item_count,
        "detail_supported_item_count": detail_supported_item_count,
        "detail_unsupported_item_count": detail_unsupported_item_count,
        "detail_supported_item_ids": detail_supported_item_ids,
        "detail_missing_allowed_item_count": detail_missing_allowed_item_count,
        "detail_supported_rows_dropped_by_item1": detail_supported_rows_dropped_by_item1,
        "detail_supported_rows_dropped_by_item9": detail_supported_rows_dropped_by_item9,
        "detail_supported_rows_kept_post_validation": detail_supported_rows_kept_post_validation,
        "detail_module3_soft_support_count": detail_module3_soft_support_count,
        "detail_module3_soft_support_item_ids": detail_module3_soft_support_item_ids,
        "detail_item14_worthlessness_hint_applied": detail_item14_worthlessness_hint_applied,
        "detail_item14_cross_scope_injected": item14_cross_scope_injected,
        "detail_item14_latent_support_applied": detail_item14_latent_support_applied,
        "detail_item21_mild_direct_keep_applied": detail_item21_mild_direct_keep_applied,
        "detail_item21_direct_denial_blocked": detail_item21_direct_denial_blocked,
        "detail_item18_change_signal_match": detail_item18_change_signal_match,
        "detail_item18_change_signal_rejected": detail_item18_change_signal_rejected,
        "item9_direct_match": bool(item9_direct_match),
        "item9_passive_risk_match": bool(item9_passive_risk_match),
        "item9_routed_risk_recovery_applied": bool(item9_routed_risk_recovery_applied),
        "opportunistic_called": opportunistic_called,
        "opportunistic_skipped_on_risk": opportunistic_skipped_on_risk,
        "opportunistic_shortlist_called": opportunistic_shortlist_called,
        "opportunistic_shortlist_parse_ok": opportunistic_shortlist_parse_ok,
        "opportunistic_has_strong_offtarget_signal": opportunistic_has_strong_offtarget_signal,
        "opportunistic_candidate_item_ids": opportunistic_candidate_item_ids,
        "opportunistic_score_called": opportunistic_score_called,
        "opportunistic_score_parse_ok": opportunistic_score_parse_ok,
        "opportunistic_parse_ok": opportunistic_parse_ok,
        "opportunistic_raw_items_count": opportunistic_raw_items_count,
        "opportunistic_kept_items_count": opportunistic_kept_items_count,
        "opportunistic_dropped_weak_count": opportunistic_dropped_weak_count,
        "opportunistic_salvage_used": opportunistic_salvage_used,
        "opportunistic_item_ids": opportunistic_item_ids,
        "opportunistic_scored_item_count": opportunistic_scored_item_count,
        "opportunistic_supported_item_count": opportunistic_supported_item_count,
        "opportunistic_unsupported_item_count": opportunistic_unsupported_item_count,
        "opportunistic_supported_item_ids": opportunistic_supported_item_ids,
        "opportunistic_missing_item_count": opportunistic_missing_item_count,
        "raw_nonempty": raw_nonempty,
        "json_parse_ok": json_parse_ok,
        "parse_error_kind": parse_error_kind,
        "parse_error_message": parse_error_message,
        "parse_error_line": parse_error_line,
        "parse_error_column": parse_error_column,
        "parse_error_position": parse_error_position,
        "parse_balance": parse_balance,
        "raw_items_count": raw_items_count,
        "kept_items_count": len(evidence_records),
        "drop_unknown_item_count": dropped_unknown,
        "drop_invalid_range_count": dropped_invalid,
        "prefilter_count": len(lexical_prefilter),
        "llm_on_lexical_hit": llm_on_lexical_hit,
        "extractor_min_records_target": extractor_min_records_target,
        "llm_called": llm_called,
        "key_alias_used_count": key_alias_used_count,
        "schema_coerce_used_count": schema_coerce_used_count,
        "symptom_name_normalized_count": symptom_name_normalized_count,
        "item1_gate_kept_count": item1_gate_kept_count,
        "item1_gate_dropped_count": item1_gate_dropped_count,
        "item1_gate_soft_clamped_count": item1_gate_soft_clamped_count,
        "precision_gate_dropped_count": precision_gate_dropped_count,
        "precision_gate_soft_clamped_count": precision_gate_soft_clamped_count,
        "precision_gate_item_counts": precision_gate_item_counts,
        "fallback_used": bool(fallback_records),
        "salvage_used": salvage_used,
        "salvage_items_count": salvage_items_count,
        "raw_extractor_payload": raw_payload_logged,
        "latest_message": latest_message if raw_payload_logged else "",
        "empty_streak": empty_streak,
        "has_new_persona_input": True,
    }
    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["extract_likelihoods"] = trace_payload
    turn_trace["extract_evidence"] = trace_payload

    denied_items_this_turn: List[int] = []
    if clear_no_symptom_skip:
        _denied_item_id = int(target_spec["target_item_id"])
        if 1 <= _denied_item_id <= 21:
            denied_items_this_turn.append(_denied_item_id)

    summary = (
        f"{state.get('specialist_debug', '')} | evidence_count={len(evidence_records)}"
        if state.get("specialist_debug")
        else f"Evidence extraction: count={len(evidence_records)}"
    )

    return {
        "latest_turn_likelihoods": likelihood_rows,
        "latest_turn_evidence": evidence_records,
        "evidence_log": evidence_records,
        "specialist_debug": summary,
        "turn_trace": turn_trace,
        "failure_counters": counters,
        "empty_evidence_streak": empty_streak,
        "denied_item_ids": denied_items_this_turn,
    }


def extract_likelihoods(state: AgentState) -> Dict:
    has_new_persona_input = bool(state.get("has_new_persona_input", False))
    turn_obj = state.get("turn")
    turn = int(getattr(turn_obj, "turn_id", int(state.get("turn_index", 0)) or 1))
    latest_message = str(getattr(turn_obj, "latest_text_raw", "") or "")

    node_name = str(state.get("active_node", "cognitive"))
    if node_name not in {"somatic", "cognitive", "risk"}:
        node_name = "cognitive"

    if not has_new_persona_input:
        turn_trace = dict(state.get("turn_trace", {}))
        trace_payload = {
            "turn": turn,
            "extractor_version": _extractor_version(),
            "source": "skip_no_new_persona",
            "kept_items_count": 0,
            "empty_streak": int(state.get("empty_evidence_streak", 0)),
            "has_new_persona_input": False,
        }
        turn_trace["extract_likelihoods"] = trace_payload
        turn_trace["extract_evidence"] = trace_payload
        return {
            "latest_turn_likelihoods": [],
            "latest_turn_evidence": [],
            "specialist_debug": "Evidence extraction: waiting for persona input",
            "turn_trace": turn_trace,
        }

    if _extractor_version() == "v2":
        return _extract_likelihoods_v2(
            state,
            turn=turn,
            latest_message=latest_message,
            node_name=node_name,
        )

    evidence_records: List[EvidenceRecord] = []
    raw_nonempty = False
    json_parse_ok = False
    raw_items_count = 0
    dropped_unknown = 0
    dropped_invalid = 0
    source = "llm_extractor"
    counters = dict(state.get("failure_counters", {}))
    lexical_prefilter: List[EvidenceRecord] = []
    clear_no_symptom_skip = False
    salvage_used = False
    salvage_items_count = 0
    key_alias_used_count = 0
    schema_coerce_used_count = 0
    symptom_name_normalized_count = 0
    item1_gate_kept_count = 0
    item1_gate_dropped_count = 0
    item1_gate_soft_clamped_count = 0
    precision_gate_dropped_count = 0
    precision_gate_soft_clamped_count = 0
    precision_gate_item_counts: Dict[str, Dict[str, int]] = {}
    parse_error_kind = ""
    parse_error_message = ""
    parse_error_line = 0
    parse_error_column = 0
    parse_error_position = 0
    parse_balance: Dict[str, Any] = {}
    llm_called = False
    llm_raw_text = ""
    raw_payload_logged = ""
    llm_on_lexical_hit = _env_bool("EVIDENCE_LLM_ON_LEXICAL_HIT", "0")
    key_aliases_enabled = _env_bool("EXTRACTOR_JSON_KEY_ALIASES", "1")
    strict_schema_coerce = _env_bool("EXTRACTOR_STRICT_SCHEMA_COERCE", "1")
    item1_strict_gate = _env_bool("EXTRACT_ITEM1_STRICT_GATE", "0")
    item1_weak_max_conf = _clamp(_env_float("EXTRACT_ITEM1_WEAK_MAX_CONF", 0.55), 0.0, 1.0)
    item1_weak_max_intensity = _clamp(_env_float("EXTRACT_ITEM1_WEAK_MAX_INTENSITY", 1.5), 0.0, 3.0)
    extractor_min_records_target = max(1, int(os.getenv("EXTRACTOR_MIN_RECORDS_TARGET", "1")))

    if latest_message.strip():
        lexical_prefilter = _fallback_evidence_from_text(node_name, turn, latest_message)
        lexical_prefilter, dropped_count, soft_clamped_count, item_counts = _apply_precision_gate_batch(
            lexical_prefilter,
            latest_message=latest_message,
        )
        precision_gate_dropped_count += int(dropped_count)
        precision_gate_soft_clamped_count += int(soft_clamped_count)
        _merge_precision_gate_item_counts(precision_gate_item_counts, item_counts)
        should_skip_llm = len(lexical_prefilter) >= extractor_min_records_target and not llm_on_lexical_hit
        if should_skip_llm:
            evidence_records = lexical_prefilter
            source = "lexical_prefilter"
        else:
            prompt = get_prompt("evidence_extraction").format(
                node_name=node_name,
                recent_context=_recent_context(state) or "none",
                latest_message=latest_message,
            )
            try:
                llm_called = True
                llm = get_extractor_llm()
                raw = llm.invoke([("system", prompt)]).content
                raw_text = str(raw)
                llm_raw_text = raw_text
                raw_nonempty = bool(raw_text.strip())
                if not raw_nonempty:
                    counters = bump_failure_counter(counters, "extract_llm_empty_payload")
                    source = "llm_extractor_empty_payload"
                parsed, json_parse_ok, parse_diagnostics = _parse_json_payload(raw_text)
                parse_error_kind = str(parse_diagnostics.get("error_kind", "") or "")
                parse_error_message = str(parse_diagnostics.get("error_message", "") or "")
                parse_error_line = int(parse_diagnostics.get("error_line", 0) or 0)
                parse_error_column = int(parse_diagnostics.get("error_column", 0) or 0)
                parse_error_position = int(parse_diagnostics.get("error_position", 0) or 0)
                parse_balance = {
                    "brace_open": int(parse_diagnostics.get("brace_open", 0) or 0),
                    "brace_close": int(parse_diagnostics.get("brace_close", 0) or 0),
                    "bracket_open": int(parse_diagnostics.get("bracket_open", 0) or 0),
                    "bracket_close": int(parse_diagnostics.get("bracket_close", 0) or 0),
                    "double_quote_count": int(parse_diagnostics.get("double_quote_count", 0) or 0),
                    "unmatched_double_quote": bool(parse_diagnostics.get("unmatched_double_quote", False)),
                }
                items, schema_payload_coerce = _payload_items(parsed)
                schema_coerce_used_count += int(schema_payload_coerce)
                if schema_payload_coerce > 0:
                    counters = bump_failure_counter(counters, "extract_schema_coerce_used", amount=schema_payload_coerce)

                if raw_nonempty and not items:
                    salvage_items = _salvage_items_from_text(raw_text)
                    if salvage_items:
                        items = salvage_items
                        salvage_used = True
                        salvage_items_count = len(salvage_items)
                        source = "llm_salvage"
                        counters = bump_failure_counter(counters, "extract_salvage_used")
                        counters = bump_failure_counter(
                            counters, "extract_salvage_kept_items", amount=salvage_items_count
                        )

                raw_items_count = len(items) if isinstance(items, list) else 0
                if isinstance(items, list):
                    for raw_item in items:
                        if not isinstance(raw_item, dict):
                            dropped_invalid += 1
                            continue
                        normalized_item, alias_hits = _normalize_item_keys(
                            raw_item,
                            key_aliases_enabled=key_aliases_enabled,
                        )
                        key_alias_used_count += int(alias_hits)
                        if alias_hits > 0:
                            counters = bump_failure_counter(counters, "extract_key_alias_used", amount=alias_hits)

                        normalized_item, schema_hits = _coerce_schema_defaults(
                            normalized_item,
                            strict_schema_coerce=strict_schema_coerce,
                        )
                        schema_coerce_used_count += int(schema_hits)
                        if schema_hits > 0:
                            counters = bump_failure_counter(counters, "extract_schema_coerce_used", amount=schema_hits)

                        symptom_name = str(normalized_item.get("symptom_name", "")).strip()
                        resolved_item_id = _coerce_item_id(normalized_item.get("item_id"), symptom_name)
                        if resolved_item_id is None:
                            dropped_unknown += 1
                            continue
                        canonical_symptom_name, normalized_symptom = _canonicalize_symptom_name(
                            resolved_item_id,
                            symptom_name,
                        )
                        normalized_item["symptom_name"] = canonical_symptom_name
                        if normalized_symptom:
                            symptom_name_normalized_count += 1
                            counters = bump_failure_counter(counters, "extract_symptom_name_normalized")
                        if "item_id" not in normalized_item:
                            normalized_item["item_id"] = resolved_item_id

                        if not _number_in_range(normalized_item.get("intensity"), 0.0, 3.0):
                            dropped_invalid += 1
                            continue
                        if not _number_in_range(normalized_item.get("confidence"), 0.0, 1.0):
                            dropped_invalid += 1
                            continue
                        record = _coerce_evidence_record(node_name, turn, normalized_item, latest_message)
                        if record is not None:
                            if _is_item1_llm_candidate(record):
                                gated_record, gate_action = _apply_item1_gate(
                                    record,
                                    latest_message=latest_message,
                                    strict_gate=item1_strict_gate,
                                    weak_max_conf=item1_weak_max_conf,
                                    weak_max_intensity=item1_weak_max_intensity,
                                )
                                if gate_action == "dropped":
                                    item1_gate_dropped_count += 1
                                    continue
                                if gate_action == "soft_clamped":
                                    item1_gate_soft_clamped_count += 1
                                else:
                                    item1_gate_kept_count += 1
                                record = gated_record
                            if record is not None:
                                gated_record, precision_action = _apply_precision_gate(
                                    record,
                                    latest_message=latest_message,
                                )
                                if precision_action == "dropped":
                                    precision_gate_dropped_count += 1
                                    _merge_precision_gate_counts(
                                        precision_gate_item_counts,
                                        item_id=int(record.item_id),
                                        action=precision_action,
                                    )
                                    continue
                                if precision_action == "soft_clamped":
                                    precision_gate_soft_clamped_count += 1
                                    _merge_precision_gate_counts(
                                        precision_gate_item_counts,
                                        item_id=int(record.item_id),
                                        action=precision_action,
                                    )
                                record = gated_record
                            evidence_records.append(record)
                else:
                    source = "llm_extractor_non_list_payload"

            except LLMBudgetExceeded:
                raise
            except Exception as exc:
                source = "llm_extractor_error"
                counters = bump_failure_counter(counters, "extract_llm_call_fail")
                error_text = str(exc).strip()
                parse_error_kind = "llm_timeout" if "timed out" in error_text.lower() else "llm_call_failed"
                parse_error_message = error_text[:300]
    else:
        source = "skip_empty_message"

    if dropped_unknown > 0:
        counters = bump_failure_counter(counters, "extract_item_map_fail", amount=dropped_unknown)

    fallback_records: List[EvidenceRecord] = []
    if not evidence_records and latest_message.strip():
        fallback_records = _fallback_evidence_from_text(node_name, turn, latest_message)
        fallback_records, dropped_count, soft_clamped_count, item_counts = _apply_precision_gate_batch(
            fallback_records,
            latest_message=latest_message,
        )
        precision_gate_dropped_count += int(dropped_count)
        precision_gate_soft_clamped_count += int(soft_clamped_count)
        _merge_precision_gate_item_counts(precision_gate_item_counts, item_counts)
        if fallback_records:
            evidence_records = fallback_records
            source = "lexical_fallback"

    if llm_called and raw_nonempty and not evidence_records:
        counters = bump_failure_counter(counters, "extract_json_parse_fail")
        raw_payload_logged = llm_raw_text

    if not evidence_records:
        counters = bump_failure_counter(counters, "extract_empty")
        empty_streak = int(state.get("empty_evidence_streak", 0)) + 1
    else:
        empty_streak = 0

    likelihood_rows: List[LikelihoodEvidence] = []
    for record in evidence_records:
        method = str(record.method or "llm_extractor")
        likelihood_rows.append(
            LikelihoodEvidence(
                item_id=int(record.item_id),
                likelihood=_likelihood_from_record(record),
                spans=[record.evidence_text],
                extract_confidence=float(record.confidence),
                extract_intensity=float(record.intensity),
                evidence_type=method,
                symptom_name=str(record.symptom_name),
                direction=str(record.direction),
                evidence_id=_evidence_id(record),
                method_weight_hint=float(METHOD_WEIGHT_HINTS.get(method, 0.50)),
                precision_gate_action=str(getattr(record, "precision_gate_action", "kept") or "kept"),
                support_increment_blocked=bool(getattr(record, "support_increment_blocked", False)),
            )
        )

    trace_payload = {
        "turn": turn,
        "source": source,
        "extractor_version": "v1",
        "raw_nonempty": raw_nonempty,
        "json_parse_ok": json_parse_ok,
        "parse_error_kind": parse_error_kind,
        "parse_error_message": parse_error_message,
        "parse_error_line": parse_error_line,
        "parse_error_column": parse_error_column,
        "parse_error_position": parse_error_position,
        "parse_balance": parse_balance,
        "raw_items_count": raw_items_count,
        "kept_items_count": len(evidence_records),
        "drop_unknown_item_count": dropped_unknown,
        "drop_invalid_range_count": dropped_invalid,
        "prefilter_count": len(lexical_prefilter),
        "llm_on_lexical_hit": llm_on_lexical_hit,
        "extractor_min_records_target": extractor_min_records_target,
        "llm_called": llm_called,
        "key_alias_used_count": key_alias_used_count,
        "schema_coerce_used_count": schema_coerce_used_count,
        "symptom_name_normalized_count": symptom_name_normalized_count,
        "item1_gate_kept_count": item1_gate_kept_count,
        "item1_gate_dropped_count": item1_gate_dropped_count,
        "item1_gate_soft_clamped_count": item1_gate_soft_clamped_count,
        "precision_gate_dropped_count": precision_gate_dropped_count,
        "precision_gate_soft_clamped_count": precision_gate_soft_clamped_count,
        "precision_gate_item_counts": precision_gate_item_counts,
        "fallback_used": bool(fallback_records),
        "salvage_used": salvage_used,
        "salvage_items_count": salvage_items_count,
        "raw_extractor_payload": raw_payload_logged,
        "latest_message": latest_message if raw_payload_logged else "",
        "empty_streak": empty_streak,
        "has_new_persona_input": True,
    }
    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["extract_likelihoods"] = trace_payload
    turn_trace["extract_evidence"] = trace_payload

    denied_items_this_turn: List[int] = []
    if clear_no_symptom_skip:
        _denied_item_id = int(target_spec["target_item_id"])
        if 1 <= _denied_item_id <= 21:
            denied_items_this_turn.append(_denied_item_id)

    summary = (
        f"{state.get('specialist_debug', '')} | evidence_count={len(evidence_records)}"
        if state.get("specialist_debug")
        else f"Evidence extraction: count={len(evidence_records)}"
    )

    return {
        "latest_turn_likelihoods": likelihood_rows,
        "latest_turn_evidence": evidence_records,
        "evidence_log": evidence_records,
        "specialist_debug": summary,
        "turn_trace": turn_trace,
        "failure_counters": counters,
        "empty_evidence_streak": empty_streak,
        "denied_item_ids": denied_items_this_turn,
    }


# Backward compatibility for old imports.
def extract_evidence(state: AgentState) -> Dict:
    return extract_likelihoods(state)
