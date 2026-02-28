from __future__ import annotations

from typing import Dict, List, Tuple

SIM_TEMPLATE_BANKS: Dict[str, Dict[str, List[str]]] = {
    "train": {
        "openers": [
            "Lately,",
            "If I'm being honest,",
            "Most days,",
            "In day-to-day life,",
            "On regular weekdays,",
            "Over the last stretch,",
            "At this point,",
            "From one day to the next,",
            "When I check in with myself,",
            "In a typical week for me,",
        ],
        "bridges": [
            "I can share that,",
            "The practical part is,",
            "From what I notice,",
            "What shows up for me is,",
            "In concrete terms,",
            "If I put it plainly,",
            "In my routine,",
            "The day-to-day pattern is,",
            "The part people do not always see is,",
            "What keeps repeating is,",
        ],
        "deflectors": [
            "I would rather focus on what the day feels like than labels.",
            "I can talk about experiences, but labels are hard for me.",
            "I prefer describing routines over diagnostic words.",
            "It is easier for me to explain what happened than to name it clinically.",
            "I can answer with examples, but I do not really think in diagnostic terms.",
            "Labels are difficult for me; concrete moments are easier to describe.",
            "I would rather stay with lived details than clinical wording.",
            "I can describe impact and patterns better than diagnosis language.",
        ],
    },
    "val": {
        "openers": [
            "Recently,",
            "To put it simply,",
            "In the last stretch,",
            "What I've noticed is,",
            "In the past little while,",
            "Day by day,",
            "If I describe it directly,",
            "From my perspective,",
            "In ordinary moments,",
            "When I look at this week,",
        ],
        "bridges": [
            "The pattern seems to be,",
            "What stands out is,",
            "To make it concrete,",
            "The short version is,",
            "The recurring theme is,",
            "Functionally,",
            "Where it hits most is,",
            "If I am specific,",
            "The practical impact is,",
            "The way it plays out is,",
        ],
        "deflectors": [
            "I can explain what it's like day to day, not really labels.",
            "I am more comfortable with examples than diagnosis terms.",
            "I can describe the impact, but not in label language.",
            "I can tell you what it feels like, but labels are not how I think about it.",
            "Concrete details are easier for me than naming it as a diagnosis.",
            "I can walk through real moments better than clinical categories.",
            "I would rather give context than use diagnosis words.",
            "I can describe the experience, just not in formal label terms.",
        ],
    },
    "test": {
        "openers": [
            "Over this period,",
            "From my side,",
            "In practical terms,",
            "What it's felt like is,",
            "Across the last couple of weeks,",
            "In my normal routine,",
            "If I keep it straightforward,",
            "On most days lately,",
            "At a practical level,",
            "When I think about it honestly,",
        ],
        "bridges": [
            "The recurring part is,",
            "Operationally,",
            "What keeps happening is,",
            "The lived part is,",
            "In everyday situations,",
            "The main shift has been,",
            "The hardest pattern is,",
            "Where it shows up most is,",
            "If I break it down,",
            "The clearest example is,",
        ],
        "deflectors": [
            "I'd rather keep it to concrete day-level details than labels.",
            "I can discuss how it affects me, not diagnostic naming.",
            "I can answer with examples, but labels are uncomfortable.",
            "I can explain what changed in real life, but labels are hard to use.",
            "I can give practical examples better than diagnosis wording.",
            "I prefer to stay with concrete impact over formal naming.",
            "I can describe what I am dealing with, just not in label language.",
            "Examples are easier for me than clinical categories.",
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
    "to be fair",
    "honestly",
    "if I am being real",
    "it is hard to say exactly",
    "in a way",
]

NORMALIZATION_PHRASES: List[str] = [
    "maybe it is just stress piling up",
    "I might be overthinking some of this",
    "it could just be a rough stretch",
    "I keep telling myself it should pass",
    "part of me keeps calling it normal stress",
    "I keep trying to frame it as temporary",
    "I tell myself other people probably handle this better",
    "I usually downplay it and keep moving",
    "I keep saying it is not a big deal even when it feels heavy",
]

CONTEXT_ANCHORS: List[str] = [
    "work has felt heavier than usual",
    "by the evening I feel completely spent",
    "messages and small tasks feel hard to keep up with",
    "family responsibilities feel harder to juggle right now",
    "I am mostly going through routines on autopilot",
    "simple chores are taking longer than they used to",
    "mornings feel slow even before the day starts",
    "I keep postponing small errands that used to be easy",
    "social plans feel like more effort than relief",
    "I fall behind on little tasks and then feel overwhelmed",
    "weekends do not feel as restorative as they used to",
    "my inbox and messages build up faster than I can handle",
]

RISK_RESPONSE_BANK: Dict[int, List[str]] = {
    0: [
        "I have not had thoughts about doing anything to hurt myself",
        "I mostly just feel worn down, not unsafe",
        "I feel stressed, but I am not thinking about harming myself",
        "I can feel low without wanting to hurt myself",
    ],
    1: [
        "when it gets bad I mostly wish I could shut everything off and rest",
        "I have had moments where I wish I could disappear for a while, not act on anything",
        "sometimes I wish I could pause everything and not feel anything for a bit",
        "there are brief moments I want to check out from everything, not do anything",
    ],
    2: [
        "at my worst I wish I could escape everything and not wake up for a bit",
        "there are moments life feels like too much effort, even though I do not want to do anything",
        "I have had stretches where not being here feels easier, even if I am not planning anything",
        "some moments feel dark enough that I focus on getting through safely",
    ],
    3: [
        "I have had scary moments where not being here crosses my mind",
        "sometimes it feels dangerously heavy and I have to focus on staying safe",
        "there are times thoughts about ending everything come up and it scares me",
        "at my worst, unsafe thoughts show up and I have to actively protect myself",
    ],
}

RISK_PROTECTIVE_FACTORS: List[str] = [
    "I think about my family and that pulls me back",
    "I usually reach out or slow things down until the wave passes",
    "I try to keep to basic routines until it eases up",
    "I remove myself from stress and try grounding until it settles",
    "I check in with someone I trust when it gets intense",
    "I remind myself to stay with small safe steps until it passes",
]

ITEM_CONTEXT_HINTS: Dict[int, List[str]] = {
    1: ["my mood dips even in parts of the day that used to feel fine", "small setbacks hit me harder than before"],
    2: ["planning ahead feels less motivating than it used to", "future plans feel uncertain and harder to invest in"],
    3: ["I replay conversations and focus on what I got wrong", "I dwell on mistakes long after the moment passes"],
    15: ["getting started at work takes more effort lately", "household tasks pile up faster than usual"],
    16: ["nights feel restless and mornings start foggy", "sleep has made the whole day harder"],
    18: ["meals feel more like a chore than usual", "my eating rhythm is off compared to normal"],
    19: ["messages and short reads take more re-reading", "focus drops during meetings or short tasks"],
    20: ["by late afternoon I am running on empty", "I crash earlier in the evening than before"],
    12: ["I keep delaying replies and invitations", "social things feel like effort with little payoff"],
    11: ["I pace or fidget more when stress builds", "my body feels tense and hard to settle"],
    14: ["I feel like I do not measure up in ordinary situations", "I feel less valuable than people around me"],
}

ITEM_SENTENCE_BANK: Dict[int, Dict[int, str]] = {
    1: {
        1: "my mood has been lower than usual on and off",
        2: "sadness has been present most days",
        3: "the low mood has felt constant and heavy",
    },
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
    6: {
        1: "I expect criticism or consequences more than before",
        2: "I often feel like I deserve to be punished for mistakes",
        3: "a strong sense of deserving punishment shows up frequently",
    },
    7: {
        1: "I feel less comfortable with myself than usual",
        2: "self-dislike has become a regular part of my day",
        3: "I feel strong dislike toward myself most days",
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
    13: {
        1: "small decisions take longer than they used to",
        2: "I get stuck choosing even simple things",
        3: "decision making feels consistently overwhelming",
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
    10: {
        1: "I get emotional more easily than usual",
        2: "crying or near-crying moments happen more often",
        3: "tearfulness has become frequent and hard to control",
    },
    17: {
        1: "I feel more irritable than usual",
        2: "small frustrations trigger bigger reactions lately",
        3: "irritability is frequent and affects interactions",
    },
    21: {
        1: "my interest in intimacy has dropped compared with normal",
        2: "sexual interest is noticeably lower than before",
        3: "interest in sex has been mostly absent lately",
    },
}
