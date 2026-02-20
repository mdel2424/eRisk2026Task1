from __future__ import annotations

import re
from typing import Dict, List, Tuple

from core.state import BDI_ITEM_NAMES

SIM_TEMPLATE_BANKS: Dict[str, Dict[str, List[str]]] = {
    "train": {
        "openers": [
            "Lately,",
            "If I'm being honest,",
            "Most days,",
            "In day-to-day life,",
        ],
        "bridges": [
            "I can share that,",
            "The practical part is,",
            "From what I notice,",
            "What shows up for me is,",
        ],
        "deflectors": [
            "I would rather focus on what the day feels like than labels.",
            "I can talk about experiences, but labels are hard for me.",
            "I prefer describing routines over diagnostic words.",
        ],
    },
    "val": {
        "openers": [
            "Recently,",
            "To put it simply,",
            "In the last stretch,",
            "What I've noticed is,",
        ],
        "bridges": [
            "The pattern seems to be,",
            "What stands out is,",
            "In concrete terms,",
            "The short version is,",
        ],
        "deflectors": [
            "I can explain what it's like day to day, not really labels.",
            "I am more comfortable with examples than diagnosis terms.",
            "I can describe the impact, but not in label language.",
        ],
    },
    "test": {
        "openers": [
            "Over this period,",
            "From my side,",
            "In practical terms,",
            "What it's felt like is,",
        ],
        "bridges": [
            "The recurring part is,",
            "Operationally,",
            "What keeps happening is,",
            "The lived part is,",
        ],
        "deflectors": [
            "I'd rather keep it to concrete day-level details than labels.",
            "I can discuss how it affects me, not diagnostic naming.",
            "I can answer with examples, but labels are uncomfortable.",
        ],
    },
}

QUESTION_KEYWORDS_TO_ITEMS: List[Tuple[List[str], int]] = [
    (["sleep", "rest", "night", "insomnia"], 16),
    (["energy", "fatigue", "tired", "drained"], 20),
    (["appetite", "eat", "meal", "weight"], 18),
    (["focus", "concentrate", "distracted"], 19),
    (["future", "hope", "tomorrow"], 2),
    (["guilt", "blame", "failure", "worthless", "worth"], 5),
    (["interest", "enjoy", "pleasure"], 4),
    (["safe", "harm", "suicide", "live"], 9),
]

DIRECT_DIAGNOSIS_TOKENS = [
    "depressed",
    "diagnosis",
    "mental illness",
    "are you okay",
    "do you have depression",
]

ITEM_SENTENCE_BANK: Dict[int, Dict[int, str]] = {
    2: {
        1: "it is harder to feel hopeful about what is coming next",
        2: "the future feels narrow and difficult to trust",
        3: "I struggle to imagine things improving in a meaningful way",
    },
    3: {
        1: "I keep replaying recent mistakes more than I used to",
        2: "I carry a persistent sense that I am falling short",
        3: "I feel like I am failing across areas that used to feel manageable",
    },
    4: {
        1: "activities feel a bit flatter than before",
        2: "things I used to enjoy feel mostly mechanical now",
        3: "even normally meaningful activities feel emotionally muted",
    },
    5: {
        1: "I tend to blame myself quickly when things go wrong",
        2: "guilt shows up repeatedly, even for small issues",
        3: "guilt has become one of the loudest parts of my day",
    },
    8: {
        1: "my inner self-talk has gotten harsher lately",
        2: "I stay stuck in self-critical loops for long stretches",
        3: "self-criticism is persistent and hard to interrupt",
    },
    9: {
        1: "when things are very heavy, thoughts can get scary",
        2: "I have moments where staying safe takes active effort",
        3: "there are episodes where thoughts about not being here appear",
    },
    11: {
        1: "I feel physically restless more often",
        2: "agitation shows up and makes it hard to settle",
        3: "my body feels keyed up and difficult to calm",
    },
    12: {
        1: "I withdraw socially more than usual",
        2: "I avoid people because interaction feels taxing",
        3: "I isolate even when I know connection might help",
    },
    14: {
        1: "I sometimes feel like a burden",
        2: "worthlessness thoughts come up regularly",
        3: "feeling fundamentally not enough has become frequent",
    },
    15: {
        1: "starting tasks takes more effort than usual",
        2: "energy drops make ordinary tasks harder to initiate",
        3: "it takes substantial effort to start even basic routines",
    },
    16: {
        1: "sleep has become inconsistent",
        2: "my sleep pattern is unstable and not restorative",
        3: "sleep disruption is frequent and leaves me depleted",
    },
    18: {
        1: "my appetite is less consistent than usual",
        2: "eating patterns are noticeably off",
        3: "appetite changes are strong enough to affect routines",
    },
    19: {
        1: "focus slips more easily than before",
        2: "concentration drops are frequent and disruptive",
        3: "sustained attention is very hard lately",
    },
    20: {
        1: "fatigue shows up earlier in the day",
        2: "tiredness persists through most of the day",
        3: "fatigue is constant and hard to push through",
    },
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_template_disjointness() -> Dict[str, object]:
    details: Dict[str, Dict[str, int]] = {}
    total_overlap = 0
    split_names = list(SIM_TEMPLATE_BANKS.keys())

    for category in ("openers", "bridges", "deflectors"):
        cat_details: Dict[str, int] = {}
        for idx in range(len(split_names)):
            left = split_names[idx]
            left_set = {_normalize(v) for v in SIM_TEMPLATE_BANKS[left][category]}
            for jdx in range(idx + 1, len(split_names)):
                right = split_names[jdx]
                right_set = {_normalize(v) for v in SIM_TEMPLATE_BANKS[right][category]}
                overlap = left_set.intersection(right_set)
                key = f"{left}__{right}"
                cat_details[key] = len(overlap)
                total_overlap += len(overlap)
        details[category] = cat_details

    return {
        "total_overlap": total_overlap,
        "details": details,
        "strict_pass": total_overlap == 0,
    }


def _latest_question(history: List[dict]) -> str:
    for msg in reversed(history):
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip().lower()
    return ""


def pick_target_item(question: str, bdi_scores: Dict[int, int], family: str, rng) -> int:
    for keywords, item_id in QUESTION_KEYWORDS_TO_ITEMS:
        if any(keyword in question for keyword in keywords):
            return item_id

    non_zero = [item_id for item_id, score in bdi_scores.items() if int(score) > 0]
    if non_zero:
        return int(rng.choice(non_zero))

    family_defaults = {
        "control_stressed": [16, 19, 15],
        "control_neutral": [4, 15],
        "somatic_evasive": [16, 20, 15],
        "cognitive_ruminative": [2, 3, 8],
        "mixed_moderate": [4, 15, 19],
        "functional_masked": [4, 15, 19],
        "risk_leaning": [9, 2, 14],
    }
    choices = family_defaults.get(family, [15, 4, 19])
    return int(rng.choice(choices))


def _item_sentence(item_id: int, score: int) -> str:
    clipped = max(0, min(3, int(score)))
    if clipped == 0:
        name = BDI_ITEM_NAMES.get(item_id, "general stress").lower()
        return f"that area feels mostly manageable, with occasional stress around {name}"

    bank = ITEM_SENTENCE_BANK.get(item_id)
    if bank:
        return bank.get(clipped, bank[max(bank.keys())])

    symptom = BDI_ITEM_NAMES.get(item_id, f"item {item_id}").lower()
    return f"{symptom} has been more noticeable lately"


def _intent(
    question: str,
    evasiveness: float,
    contradiction_rate: float,
    rng,
) -> str:
    if any(token in question for token in DIRECT_DIAGNOSIS_TOKENS) and rng.random() < min(0.95, evasiveness + 0.25):
        return "deflect"
    if contradiction_rate > 0.0 and rng.random() < min(0.35, contradiction_rate):
        return "partial"
    if rng.random() < max(0.15, evasiveness * 0.5):
        return "partial"
    return "disclose"


def _safe_join(chunks: List[str], limit_words: int = 38) -> str:
    text = " ".join(chunk.strip() for chunk in chunks if chunk and chunk.strip())
    words = text.split()
    if len(words) > limit_words:
        text = " ".join(words[:limit_words]).rstrip(" ,;") + "."
    if text and text[-1] not in {".", "?", "!"}:
        text += "."
    return re.sub(r"\s+", " ", text).strip()


def build_deterministic_reply(
    *,
    family: str,
    split: str,
    bdi_scores: Dict[int, int],
    behavior_params: Dict[str, float | str],
    history: List[dict],
    evasive: bool,
    rng,
) -> str:
    split_key = split if split in SIM_TEMPLATE_BANKS else "test"
    bank = SIM_TEMPLATE_BANKS[split_key]

    question = _latest_question(history)
    target_item = pick_target_item(question, bdi_scores, family, rng)
    target_score = int(bdi_scores.get(target_item, 0))

    evasiveness = float(behavior_params.get("evasiveness", 0.45))
    contradiction_rate = float(behavior_params.get("contradiction", 0.08))
    intent = _intent(question, evasiveness if evasive else 0.1, contradiction_rate, rng)

    opener = rng.choice(bank["openers"])
    bridge = rng.choice(bank["bridges"])

    if intent == "deflect":
        return _safe_join([rng.choice(bank["deflectors"]), f"{bridge} {_item_sentence(target_item, target_score)}"])

    if intent == "partial":
        softened = max(0, target_score - 1)
        return _safe_join([f"{opener} it is a bit hard to describe directly", f"{bridge} {_item_sentence(target_item, softened)}"])

    return _safe_join([f"{opener} {_item_sentence(target_item, target_score)}", f"{bridge} it has affected my routine more than usual"])


def normalize_response(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    # Keep response compact and conversational.
    if cleaned.count(".") > 2:
        parts = [part.strip() for part in cleaned.split(".") if part.strip()]
        cleaned = ". ".join(parts[:2]).strip()
        if cleaned and cleaned[-1] not in {".", "?", "!"}:
            cleaned += "."
    return cleaned
