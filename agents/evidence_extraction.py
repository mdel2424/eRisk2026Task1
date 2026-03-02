from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Tuple

from agents.evidence_lexicon import LEXICAL_EVIDENCE_CUES
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
}

METHOD_WEIGHT_HINTS = {
    "llm_extractor": 1.00,
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
)

ITEM1_WEAK_PATTERNS = (
    re.compile(r"\bbeen\s+down\b"),
    re.compile(r"\bmood\s+down\b"),
    re.compile(r"\bkind\s+of\s+flat\b"),
    re.compile(r"\bfeels?\s+flat\b"),
    re.compile(r"\bfeels?\s+numb\b"),
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
    return str(record.method).strip().lower() in {"llm_extractor", "llm_salvage"}


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

    evidence_records: List[EvidenceRecord] = []
    raw_nonempty = False
    json_parse_ok = False
    raw_items_count = 0
    dropped_unknown = 0
    dropped_invalid = 0
    source = "llm_extractor"
    counters = dict(state.get("failure_counters", {}))
    lexical_prefilter: List[EvidenceRecord] = []
    salvage_used = False
    salvage_items_count = 0
    key_alias_used_count = 0
    schema_coerce_used_count = 0
    symptom_name_normalized_count = 0
    item1_gate_kept_count = 0
    item1_gate_dropped_count = 0
    item1_gate_soft_clamped_count = 0
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
    item1_strict_gate = _env_bool("EXTRACT_ITEM1_STRICT_GATE", "1")
    item1_weak_max_conf = _clamp(_env_float("EXTRACT_ITEM1_WEAK_MAX_CONF", 0.55), 0.0, 1.0)
    item1_weak_max_intensity = _clamp(_env_float("EXTRACT_ITEM1_WEAK_MAX_INTENSITY", 1.5), 0.0, 3.0)
    extractor_min_records_target = max(1, int(os.getenv("EXTRACTOR_MIN_RECORDS_TARGET", "1")))

    if latest_message.strip():
        lexical_prefilter = _fallback_evidence_from_text(node_name, turn, latest_message)
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
                            evidence_records.append(record)
                else:
                    source = "llm_extractor_non_list_payload"

            except LLMBudgetExceeded:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Detector LLM evidence extraction failed at node '{node_name}' on turn {turn}."
                ) from exc
    else:
        source = "skip_empty_message"

    if dropped_unknown > 0:
        counters = bump_failure_counter(counters, "extract_item_map_fail", amount=dropped_unknown)

    fallback_records: List[EvidenceRecord] = []
    if not evidence_records and latest_message.strip():
        fallback_records = _fallback_evidence_from_text(node_name, turn, latest_message)
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
                evidence_type=method,
                symptom_name=str(record.symptom_name),
                direction=str(record.direction),
                evidence_id=_evidence_id(record),
                method_weight_hint=float(METHOD_WEIGHT_HINTS.get(method, 0.50)),
            )
        )

    trace_payload = {
        "turn": turn,
        "source": source,
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
    }


# Backward compatibility for old imports.
def extract_evidence(state: AgentState) -> Dict:
    return extract_likelihoods(state)
