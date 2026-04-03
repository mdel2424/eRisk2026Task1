from __future__ import annotations

from typing import Dict, List

SIM_TEMPLATE_BANK: Dict[str, List[str]] = {
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
        "I would rather focus on what the day feels like than labels.",
        "I can talk about experiences, but labels are hard for me.",
        "I prefer describing routines over diagnostic words.",
        "It is easier for me to explain what happened than to name it clinically.",
        "I can answer with examples, but I do not really think in diagnostic terms.",
        "Labels are difficult for me; concrete moments are easier to describe.",
        "I would rather stay with lived details than clinical wording.",
        "I can describe impact and patterns better than diagnosis language.",
        "I can explain what it's like day to day, not really labels.",
        "I am more comfortable with examples than diagnosis terms.",
        "I can describe the impact, but not in label language.",
        "I can tell you what it feels like, but labels are not how I think about it.",
        "Concrete details are easier for me than naming it as a diagnosis.",
        "I can walk through real moments better than clinical categories.",
        "I would rather give context than use diagnosis words.",
        "I can describe the experience, just not in formal label terms.",
        "I'd rather keep it to concrete day-level details than labels.",
        "I can discuss how it affects me, not diagnostic naming.",
        "I can answer with examples, but labels are uncomfortable.",
        "I can explain what changed in real life, but labels are hard to use.",
        "I can give practical examples better than diagnosis wording.",
        "I prefer to stay with concrete impact over formal naming.",
        "I can describe what I am dealing with, just not in label language.",
        "Examples are easier for me than clinical categories.",
    ],
}

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

CONTEXT_TAG_ANCHORS: Dict[str, List[str]] = {
    "workload": [
        "deadlines pile up faster than I can clear them",
        "my work queue keeps growing even when I stay on top of it",
        "small tasks at work take more effort than they should",
    ],
    "school": [
        "coursework and studying take up a lot more mental space lately",
        "assignments hang over me even when I try to switch off",
        "keeping up with classes and deadlines feels heavier than usual",
    ],
    "caregiving": [
        "other people's needs tend to take the front seat most days",
        "keeping everyone else afloat leaves me with less room to recover",
        "home responsibilities eat up a lot of my energy lately",
    ],
    "relationship_strain": [
        "tension at home keeps bleeding into the rest of the day",
        "relationship stress tends to sit in the background all day",
        "feeling disconnected from people close to me makes everything feel harder",
    ],
    "health_stress": [
        "physical discomfort makes it harder to tell where the stress stops",
        "health stuff has been sitting in the background a lot lately",
        "worrying about my body adds another layer to the day",
    ],
    "financial_pressure": [
        "money stress keeps humming in the background most days",
        "budget worries follow me through pretty ordinary decisions",
        "finances make everything feel less flexible than it used to",
    ],
    "social_isolation": [
        "I spend a lot more time on my own than I mean to",
        "feeling cut off from people makes the days run together",
        "even when people are around I still feel a bit on my own",
    ],
    "routine_stable": [
        "my routine is still pretty steady overall",
        "the basic shape of my days has stayed fairly consistent",
        "there's still a decent structure to most days for me",
    ],
}

NEUTRAL_CONTEXT_TAG_ANCHORS: Dict[str, List[str]] = {
    "workload": [
        "work has been busy but still manageable",
        "deadlines are there, but I can keep pace with them",
    ],
    "school": [
        "classes and coursework feel pretty manageable right now",
        "school takes time, but it still feels workable",
    ],
    "caregiving": [
        "family responsibilities feel busy but still manageable",
        "home responsibilities take time, but they feel steady enough",
    ],
    "relationship_strain": [
        "things with people close to me feel pretty stable overall",
        "relationship stress has not really been a big factor lately",
    ],
    "health_stress": [
        "health-wise things feel pretty steady right now",
        "physical stuff is not taking up much headspace lately",
    ],
    "financial_pressure": [
        "money is something I watch, but it does not feel overwhelming",
        "financially things feel stable enough day to day",
    ],
    "social_isolation": [
        "I still feel connected enough to people around me",
        "socially things feel pretty normal for me",
    ],
    "routine_stable": [
        "my routine still feels pretty balanced",
        "day-to-day structure is still working for me",
    ],
}

STYLE_OPENERS: Dict[str, List[str]] = {
    "terse_guarded": [
        "Honestly,",
        "If I keep it short,",
        "Short version,",
        "Plainly,",
    ],
    "contextual_reflective": [
        "When I think about it,",
        "If I put it in context,",
        "Looking at the last couple of weeks,",
        "What I notice most is,",
    ],
    "minimizing_practical": [
        "Practically speaking,",
        "Functionally,",
        "Day to day,",
        "In concrete terms,",
    ],
    "open_but_flat": [
        "Pretty simply,",
        "To be direct,",
        "The straightforward version is,",
        "Mostly,",
    ],
    "hedged_uncertain": [
        "I think,",
        "Maybe,",
        "It's hard to say exactly, but",
        "If I'm being real,",
    ],
}

STYLE_DEFLECTORS: Dict[str, List[str]] = {
    "terse_guarded": [
        "I'd rather stick to the practical part than labels.",
        "Labels are not really how I think about it.",
    ],
    "contextual_reflective": [
        "It's easier for me to explain the pattern than label it.",
        "I can talk through the context more easily than the label.",
    ],
    "minimizing_practical": [
        "I can explain the day-to-day impact better than the label.",
        "I usually think about what changed practically, not labels.",
    ],
    "open_but_flat": [
        "I can describe it directly, just not in diagnosis terms.",
        "Examples are easier for me than labels.",
    ],
    "hedged_uncertain": [
        "I'm not great with labels, but I can describe what it feels like.",
        "It's easier to talk about the experience than the label for me.",
    ],
}

ALL_CONTEXT_ANCHORS: List[str] = CONTEXT_ANCHORS + [
    anchor
    for anchors in CONTEXT_TAG_ANCHORS.values()
    for anchor in anchors
] + NEUTRAL_CONTEXT_ANCHORS + [
    anchor
    for anchors in NEUTRAL_CONTEXT_TAG_ANCHORS.values()
    for anchor in anchors
]

RESPONSE_QUALIFIERS: List[str] = [
    "I think",
    "I guess",
    "maybe",
    "mostly",
    "a bit",
    "to be honest",
]

BASELINE_COMPARISON_PHRASES: Dict[int, List[str]] = {
    0: [
        "compared with usual, that part feels about the same",
        "relative to my normal baseline, I haven't noticed much shift there",
    ],
    1: [
        "it's a little more noticeable than usual",
        "there's a small change compared with how I usually am",
    ],
    2: [
        "it's been more of a problem than it used to be",
        "it's definitely showing up more than it normally would",
    ],
    3: [
        "it's there most days now, not just once in a while",
        "it's become a pretty steady part of the last couple of weeks",
    ],
}

SOFT_DENIAL_PHRASES: List[str] = [
    "that part feels pretty close to normal for me",
    "I haven't noticed much change there compared with usual",
    "that's not really the part that's shifted the most",
    "that one has stayed mostly the same",
]

OPENING_SUMMARY_BANK: Dict[str, List[str]] = {
    "somatic_fatigue_sleep": [
        "the main change has been energy and sleep both feeling off lately",
        "what stands out most is how drained I feel and how uneven sleep has gotten",
    ],
    "appetite_variability": [
        "my routine around food and basic upkeep has been less steady than usual",
        "the physical side has felt less predictable, especially with appetite and day-to-day rhythm",
    ],
    "interest_withdrawal": [
        "motivation has been lower and it takes more effort to stay engaged in things",
        "I can still show up, but interest and enjoyment drop off pretty quickly",
    ],
    "irritability_tension": [
        "I have been more on edge and easier to set off than usual",
        "the biggest shift is probably how tense and irritable I feel through the day",
    ],
    "cognitive_self_eval": [
        "I have been pretty hard on myself and stuck in my own head lately",
        "a lot of it feels tied to second-guessing myself and feeling down on who I am",
    ],
    "focus_decision": [
        "it has been easier to get mentally bogged down and harder to stay on top of decisions",
        "my head feels more cluttered lately, especially with focus and small choices",
    ],
    "hopeless_risk": [
        "the bigger change is that things have felt heavier and darker lately",
        "I have had a harder time feeling hopeful or steady lately",
    ],
}

CONTROL_OPENING_SUMMARY_BANK: List[str] = [
    "mostly it has felt like stress and routine pressure more than anything severe",
    "it has been more of a general strain than one specific thing taking over",
    "the main change has been feeling a bit worn down, but still mostly steady overall",
]

CONTRASTIVE_NEGATIVE_BANK: Dict[str, List[str]] = {
    "sadness_vs_irritability": [
        "not so much sadness, more irritability and feeling worn down lately",
        "it is less sadness and more irritability, like I feel on edge and worn out",
    ],
    "interest_vs_energy": [
        "it is less that I do not care and more that everything takes extra effort lately",
        "interest is not the main issue there, it is more exhaustion and effort",
    ],
    "appetite_vs_fatigue": [
        "appetite itself is not the main thing; it is more the fatigue and routine feeling off",
        "it is less about appetite and more that my whole day feels worn down",
    ],
    "sleep_vs_focus": [
        "it is less that exact symptom and more that sleep and mental pace have both been off",
        "not so much that specific piece, more that my sleep and focus have been uneven",
    ],
}

QUALIFIED_UNSURE_PHRASES: List[str] = [
    "I haven't tracked it that closely, but it seems a little different",
    "it's hard to be exact, though I think there has been some shift",
    "I'm not completely sure, but it feels at least a bit off compared with usual",
]

MINIMIZATION_FRAGMENTS: List[str] = [
    "I keep telling myself it's probably just stress",
    "I know it sounds smaller when I say it out loud",
    "part of me still tries to brush it off as a rough patch",
    "I try to tell myself it should be manageable, even if it doesn't feel that way",
]

ITEM_CONCRETE_EXAMPLES: Dict[int, List[str]] = {
    1: [
        "I get home and just sink into the couch instead of doing much of anything",
        "even normal parts of the day feel heavier than they used to",
    ],
    4: [
        "I still pick up hobbies sometimes, but the spark drops out fast",
        "I can go through the motions of something I used to enjoy and still feel flat afterward",
    ],
    10: [
        "small things can suddenly make me tear up",
        "I'll catch myself crying over something that normally would not hit me that hard",
    ],
    11: [
        "little annoyances like noise or traffic get under my skin fast",
        "I can feel keyed up in my body even when I try to sit still",
    ],
    12: [
        "messages stack up and I keep putting off answering them",
        "I find myself turning down plans I normally would have said yes to",
    ],
    13: [
        "I can stand there over a small decision longer than makes sense",
        "simple choices take more mental energy than they should",
    ],
    14: [
        "I keep landing on the feeling that I am letting people down",
        "it is hard not to feel like I am falling short of who I should be",
    ],
    15: [
        "getting out of bed or starting the first task takes too much effort",
        "I can sit there knowing what needs to happen and still struggle to begin",
    ],
    16: [
        "I wake in the night and end up lying there for a while",
        "sleep breaks up enough that the next day feels off before it starts",
    ],
    17: [
        "small inconveniences can make me sharper with people than I mean to be",
        "I notice myself getting short over things that would not normally bother me much",
    ],
    18: [
        "some days food sounds fine and other days I barely bother with it",
        "eating has felt more inconsistent than usual lately",
    ],
    19: [
        "I can reread the same page and still not really absorb it",
        "my attention slips out from under me in the middle of simple tasks",
    ],
    20: [
        "by the afternoon it feels like I have already used up the day's energy",
        "even after rest I still feel wrung out more than usual",
    ],
    21: [
        "that side of things feels lower than it usually would",
        "interest in closeness has dropped off compared with my normal baseline",
    ],
}

CONTEXT_RESPONSE_EXAMPLES: Dict[str, List[str]] = {
    "workload": [
        "deadlines and follow-ups seem to pile up faster than I can clear them",
        "the workday can feel heavier before it even really gets going",
    ],
    "school": [
        "coursework stays in my head even when I try to switch off",
        "assignments take up more mental space than they used to",
    ],
    "caregiving": [
        "looking after everyone else can leave me with not much left for myself",
        "home responsibilities take up a lot of bandwidth before the day is even half done",
    ],
    "relationship_strain": [
        "tension at home bleeds into the rest of the day more than I want it to",
        "it is harder to feel settled when things feel strained with people close to me",
    ],
    "health_stress": [
        "physical discomfort adds another layer to how drained everything feels",
        "worrying about my body makes it harder to tell where the stress really starts",
    ],
    "financial_pressure": [
        "money worries can sit behind even ordinary decisions lately",
        "budget stress keeps humming in the background of normal days",
    ],
    "social_isolation": [
        "I can go through whole stretches of the day feeling cut off from people",
        "even when messages come in, I often do not have much in me to answer them",
    ],
    "routine_stable": [
        "the structure of the day is still there even if I feel a bit off inside it",
        "my routine is intact, even if parts of it feel less automatic than usual",
    ],
}

PARTIAL_ANSWER_BANK: Dict[str, List[str]] = {
    "tone_balance": [
        "it is less pure sadness and more that I feel worn down and irritable",
        "if I had to choose, it leans more toward irritability than outright sadness",
    ],
    "energy_interest": [
        "it is a bit of both, honestly; starting things takes more effort and I get less out of them once I do",
        "both show up; the energy is low and the interest fades quickly once I begin",
    ],
    "slowed_restless": [
        "mostly I feel heavy, but there is still a restless edge underneath it sometimes",
        "it is a mix; I can feel slowed down and still kind of keyed up at the same time",
    ],
    "appetite_variability": [
        "it has been up and down rather than clearly one direction the whole time",
        "it is not completely steady; some days are lower and some feel more normal",
    ],
}

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
        1: "I feel low sometimes, but it passes pretty quickly",
        2: "most days there's this heaviness I can't really shake off",
        3: "the low mood is constant, like I'm walking around with a cloud over my head",
    },
    2: {
        1: "sometimes I wonder if things will work out, but mostly I think they will",
        2: "I don't really see things getting better anytime soon, you know?",
        3: "honestly I can't picture things improving, it's hard to even think about the future",
    },
    3: {
        1: "I think about things that didn't go well a bit more than I used to",
        2: "I feel like I'm falling behind and it's my own fault",
        3: "everything I look back on feels like a string of mistakes I can't fix",
    },
    4: {
        1: "some things feel flatter, but I still enjoy most stuff",
        2: "I still do stuff but it feels more like a chore than something I actually want to do",
        3: "even things that should make me happy just feel empty, you know?",
    },
    5: {
        1: "I notice guilty feelings popping up a bit more than usual over small things",
        2: "guilt just shows up out of nowhere, even for stuff that isn't a big deal",
        3: "I feel guilty about everything, even things that aren't really my fault",
    },
    6: {
        1: "once in a while I worry I might get what I deserve for something I did",
        2: "I keep thinking I probably deserve whatever bad stuff happens",
        3: "there's this voice telling me I deserve to be punished for how things turned out",
    },
    7: {
        1: "I've lost a bit of confidence in myself, nothing major though",
        2: "I just don't like who I am right now, it's hard to explain",
        3: "I genuinely dislike who I've become and it's hard to admit that",
    },
    8: {
        1: "I'm a bit harder on myself than I probably should be",
        2: "I'm stuck in these loops where I tear everything I do apart",
        3: "I can't do anything without that voice telling me I'm doing it wrong",
    },
    9: {
        1: "when things are very heavy, thoughts can get scary",
        2: "I have moments where staying safe takes active effort",
        3: "there are episodes where thoughts about not being here appear",
    },
    10: {
        1: "I've noticed I get a bit more emotional than usual over small things",
        2: "I tear up at random things that wouldn't normally get to me",
        3: "I've been crying a lot and it feels way out of proportion to what's actually happening",
    },
    11: {
        1: "I've been a tiny bit more restless, nothing that really stands out",
        2: "I feel wound up, like I'm buzzing with this nervous energy I can't burn off",
        3: "I'm pacing and snapping at people, I just can't settle down",
    },
    12: {
        1: "I've turned down a few things I'd normally go to, but nothing drastic",
        2: "social stuff feels like way more effort than it's worth right now",
        3: "I just don't want to see anyone, even people I care about",
    },
    13: {
        1: "decisions take me a little longer than they used to, but I get there",
        2: "I get stuck on the simplest choices, like what to eat or what to watch",
        3: "I can't make decisions at all, I just sit there going back and forth",
    },
    14: {
        1: "every now and then I wonder if I'm pulling my weight, but it's not a big thing",
        2: "I feel like a burden to the people around me",
        3: "I genuinely feel worthless, like I don't contribute anything that matters",
    },
    15: {
        1: "it takes more effort to get going lately",
        2: "I have to force myself to do basic things, like I'm pushing through mud",
        3: "even getting out of bed is a battle, everything takes so much energy",
    },
    16: {
        1: "my sleep is a bit less consistent, but nothing too bad",
        2: "I wake up in the middle of the night and can't get back to sleep, so I'm always tired",
        3: "sleep is a mess, I'm either up all night or sleeping too much and still feeling wrecked",
    },
    17: {
        1: "I notice I get a bit snappier than usual, but I catch myself",
        2: "I'm getting annoyed way more easily, little things set me off",
        3: "I'm irritable all the time, like I'm walking around with a short fuse",
    },
    18: {
        1: "my appetite has been off in small ways, nothing major",
        2: "I'm either not eating at all or just grabbing junk because I can't be bothered",
        3: "food doesn't interest me at all, I just forget to eat and then feel worse",
    },
    19: {
        1: "my focus slips a bit more easily but I can still get things done",
        2: "I have to re-read things three times before anything actually sticks",
        3: "my concentration is shot, I can't follow a conversation without zoning out",
    },
    20: {
        1: "I get more worn down by the end of the day",
        2: "I feel like I'm running on fumes, just dragging myself through the day",
        3: "I'm exhausted from the moment I wake up, like I haven't slept at all",
    },
    21: {
        1: "that side of things feels a little lower lately, but not in a dramatic way",
        2: "I used to enjoy it but now it just feels like another thing I should be doing",
        3: "interest in that is basically gone, I can't even muster the energy to think about it",
    },
}
