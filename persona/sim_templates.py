from __future__ import annotations

from typing import Dict, List, Tuple

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

HEDGE_PHRASES: List[str] = [
    "I guess",
    "maybe",
    "I don't know",
    "I think",
    "kind of",
]

NORMALIZATION_PHRASES: List[str] = [
    "maybe it is just stress piling up",
    "I might be overthinking some of this",
    "it could just be a rough stretch",
    "I keep telling myself it should pass",
]

CONTEXT_ANCHORS: List[str] = [
    "work has felt heavier than usual",
    "by the evening I feel completely spent",
    "messages and small tasks feel hard to keep up with",
    "family responsibilities feel harder to juggle right now",
    "I am mostly going through routines on autopilot",
]

RISK_RESPONSE_BANK: Dict[int, List[str]] = {
    0: [
        "I have not had thoughts about doing anything to hurt myself",
        "I mostly just feel worn down, not unsafe",
    ],
    1: [
        "when it gets bad I mostly wish I could shut everything off and rest",
        "I have had moments where I wish I could disappear for a while, not act on anything",
    ],
    2: [
        "at my worst I wish I could escape everything and not wake up for a bit",
        "there are moments life feels like too much effort, even though I do not want to do anything",
    ],
    3: [
        "I have had scary moments where not being here crosses my mind",
        "sometimes it feels dangerously heavy and I have to focus on staying safe",
    ],
}

RISK_PROTECTIVE_FACTORS: List[str] = [
    "I think about my family and that pulls me back",
    "I usually reach out or slow things down until the wave passes",
    "I try to keep to basic routines until it eases up",
]

ITEM_CONTEXT_HINTS: Dict[int, List[str]] = {
    15: ["getting started at work takes more effort lately", "household tasks pile up faster than usual"],
    16: ["nights feel restless and mornings start foggy", "sleep has made the whole day harder"],
    18: ["meals feel more like a chore than usual", "my eating rhythm is off compared to normal"],
    19: ["messages and short reads take more re-reading", "focus drops during meetings or short tasks"],
    20: ["by late afternoon I am running on empty", "I crash earlier in the evening than before"],
    12: ["I keep delaying replies and invitations", "social things feel like effort with little payoff"],
}

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
