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
    "I know I should probably handle it better",
    "I'm probably just overthinking some of this",
    "it could just be a rough patch, I keep telling myself that",
    "I keep saying it should pass eventually",
    "part of me thinks everyone goes through this kind of thing",
    "I keep trying to tell myself it's temporary",
    "I know other people deal with worse, so I try not to dwell on it",
    "I usually just push through and try not to think about it too much",
    "I keep telling myself it's not that big a deal even when it feels heavy",
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

NEUTRAL_CONTEXT_ANCHORS: List[str] = [
    "work has been steady and manageable",
    "evenings are usually relaxing for me",
    "I keep up with messages and daily tasks without much trouble",
    "family routines feel comfortable right now",
    "my daily routine has a good rhythm to it",
    "chores get done at a normal pace",
    "mornings start at a comfortable pace",
    "I stay on top of small errands without much effort",
    "social plans still feel worth the energy",
    "little tasks get handled as they come up",
    "weekends still feel restorative",
    "I manage my inbox at a reasonable pace",
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
    1: ["even parts of the day that used to feel fine have this heaviness to them", "small setbacks hit me way harder than they used to"],
    2: ["planning ahead just feels pointless right now", "I used to get excited about the future but now I just feel blank"],
    3: ["I replay conversations in my head and focus on what I got wrong", "I dwell on old mistakes way longer than I should"],
    15: ["getting started at work takes so much more effort than it used to", "stuff around the house just piles up because I can't get going"],
    16: ["I wake up at like 3am and just lie there staring at the ceiling", "sleep has made the whole next day harder"],
    18: ["meals feel more like a chore, I just eat because I have to", "I grab whatever is closest because I can't be bothered to cook"],
    19: ["I read messages and have to go back and re-read them", "I zone out in meetings and miss whole chunks of what people say"],
    20: ["by late afternoon I'm completely spent", "I crash way earlier in the evening than I used to"],
    12: ["I keep putting off replies to messages and invitations", "being around people feels like a lot of effort for not much payoff"],
    11: ["I pace around or fidget more when stress builds up", "my body feels tense and I can't seem to relax"],
    14: ["I look around and everyone else seems to have their life sorted", "I feel like I don't measure up compared to the people around me"],
}

NEUTRAL_ITEM_CONTEXT_HINTS: Dict[int, List[str]] = {
    1: ["my mood has been pretty even day to day", "I bounce back from setbacks at a normal pace"],
    2: ["I still feel motivated when I think about upcoming plans", "future plans feel realistic and worth working toward"],
    3: ["I can usually move on from mistakes without dwelling on them", "past situations do not weigh on me much"],
    15: ["I get started on work without much delay", "household tasks stay manageable"],
    16: ["sleep has been fairly consistent for me", "I generally wake up feeling rested enough"],
    18: ["my appetite has been about normal", "meals fit into my day without much thought"],
    19: ["I can follow conversations and reading without much trouble", "focus has felt normal for me"],
    20: ["my energy holds up through most of the day", "I feel tired at reasonable times"],
    12: ["I keep up with replies and social commitments", "social things still feel worthwhile"],
    11: ["I feel physically settled most of the time", "I can sit still and relax without much restlessness"],
    14: ["I feel like I hold my own in everyday situations", "I feel reasonably confident compared to others around me"],
}

ITEM_SENTENCE_BANK: Dict[int, Dict[int, str]] = {
    1: {
        1: "I've had this low feeling hanging around more than usual",
        2: "most days there's this heaviness I can't really shake off",
        3: "the low mood is constant, like I'm walking around with a cloud over my head",
    },
    2: {
        1: "looking ahead doesn't feel as motivating as it used to",
        2: "I don't really see things getting better anytime soon, you know?",
        3: "honestly I can't picture things improving, it's hard to even think about the future",
    },
    3: {
        1: "I keep going over things I messed up more than I used to",
        2: "I feel like I'm falling behind and it's my own fault",
        3: "everything I look back on feels like a string of mistakes I can't fix",
    },
    4: {
        1: "things I used to enjoy feel kind of flat now, like I'm just going through the motions",
        2: "I still do stuff but it feels more like a chore than something I actually want to do",
        3: "even things that should make me happy just feel empty, you know?",
    },
    5: {
        1: "I catch myself feeling guilty about little things more than I should",
        2: "guilt just shows up out of nowhere, even for stuff that isn't a big deal",
        3: "I feel guilty about everything, even things that aren't really my fault",
    },
    6: {
        1: "sometimes I feel like I've got bad things coming because of my mistakes",
        2: "I keep thinking I probably deserve whatever bad stuff happens",
        3: "there's this voice telling me I deserve to be punished for how things turned out",
    },
    7: {
        1: "I'm not really happy with myself lately, if I'm honest",
        2: "I just don't like who I am right now, it's hard to explain",
        3: "I genuinely dislike who I've become and it's hard to admit that",
    },
    8: {
        1: "my inner voice has gotten pretty harsh, I beat myself up a lot",
        2: "I'm stuck in these loops where I tear everything I do apart",
        3: "I can't do anything without that voice telling me I'm doing it wrong",
    },
    9: {
        1: "when things are very heavy, thoughts can get scary",
        2: "I have moments where staying safe takes active effort",
        3: "there are episodes where thoughts about not being here appear",
    },
    10: {
        1: "I get this lump in my throat more easily than before",
        2: "I tear up at random things that wouldn't normally get to me",
        3: "I've been crying a lot and it feels way out of proportion to what's actually happening",
    },
    11: {
        1: "I've been more fidgety, I can't seem to sit still",
        2: "I feel wound up, like I'm buzzing with this nervous energy I can't burn off",
        3: "I'm pacing and snapping at people, I just can't settle down",
    },
    12: {
        1: "I've been pulling back from people more than usual",
        2: "social stuff feels like way more effort than it's worth right now",
        3: "I just don't want to see anyone, even people I care about",
    },
    13: {
        1: "little decisions take me way longer than they should",
        2: "I get stuck on the simplest choices, like what to eat or what to watch",
        3: "I can't make decisions at all, I just sit there going back and forth",
    },
    14: {
        1: "sometimes I feel like I'm just taking up space",
        2: "I feel like a burden to the people around me",
        3: "I genuinely feel worthless, like I don't contribute anything that matters",
    },
    15: {
        1: "getting started on things takes way more effort than it used to",
        2: "I have to force myself to do basic things, like I'm pushing through mud",
        3: "even getting out of bed is a battle, everything takes so much energy",
    },
    16: {
        1: "my sleep has been all over the place lately",
        2: "I wake up in the middle of the night and can't get back to sleep, so I'm always tired",
        3: "sleep is a mess, I'm either up all night or sleeping too much and still feeling wrecked",
    },
    17: {
        1: "I snap at people over little things, which isn't really like me",
        2: "I'm getting annoyed way more easily, little things set me off",
        3: "I'm irritable all the time, like I'm walking around with a short fuse",
    },
    18: {
        1: "my appetite has been off, I eat because I should not because I want to",
        2: "I'm either not eating at all or just grabbing junk because I can't be bothered",
        3: "food doesn't interest me at all, I just forget to eat and then feel worse",
    },
    19: {
        1: "I can't focus like I used to, my mind just wanders off",
        2: "I have to re-read things three times before anything actually sticks",
        3: "my concentration is shot, I can't follow a conversation without zoning out",
    },
    20: {
        1: "I'm tired earlier in the day than I used to be",
        2: "I feel like I'm running on fumes, just dragging myself through the day",
        3: "I'm exhausted from the moment I wake up, like I haven't slept at all",
    },
    21: {
        1: "that side of things has dropped off, I just don't have the interest",
        2: "I used to enjoy it but now it just feels like another thing I should be doing",
        3: "interest in that is basically gone, I can't even muster the energy to think about it",
    },
}
