"""Geo's Summer Reset plan — structured data + schedule helpers.

This is the single source of truth the assistant reads for the weight-loss
features (weigh-in trend, workout-of-the-day, briefing, progress report,
reminders). Editing the plan? Change it here; every feature follows.
Dates are plain YYYY-MM-DD; all helpers take an optional `on` date
(defaults to today) so they're easy to unit-test.
"""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

RESET_DATE = date(2026, 7, 13)
START_WEIGHT = 180.0
GOAL_WEIGHT = 147.0
HEIGHT = "5'6\""

# daily nutrition targets (Phase 1)
MACROS = {"calories": 1500, "protein": 180, "carbs": 100, "fat": 40}
REFEED_CALORIES = 1900
REFEED_EVERY_DAYS = 14          # every 2 weeks from RESET_DATE

PHASES = [
    {"name": "Phase 1 — The Reset",      "start": date(2026, 7, 13), "end": date(2026, 8, 24),
     "note": "1,500 cal/day, -1.5 to -2 lbs/week. 4 lifts + 2 cardio + 1 rest."},
    {"name": "Phase 2 — Extended Cut",   "start": date(2026, 8, 25), "end": date(2026, 9, 21),
     "note": "Reassess on the 7-day trend; adjust one variable at a time."},
    {"name": "Phase 3 — The Reward",     "start": date(2026, 9, 22), "end": None,
     "note": "Within ~10 lbs of goal: calories up, training toward building."},
]

# weekday() -> (focus label, list of exercises). Mon=0 .. Sun=6
CARDIO = ["25–30 min run, Zone 2 conversational pace",
          "(legs wrecked? a brisk walk counts)"]

SPLIT: Dict[int, Dict] = {
    0: {"focus": "Back & Posterior Chain", "kind": "lift", "exercises": [
        "Pull-Ups — 3 × max reps",
        "T-Bar Row — 20 warm-up, 15, 2×10–12 to failure",
        "Seated Cable Row — 20 warm-up, 15, 2×12 to failure",
        "Wide Grip Lat Pulldown — 2×12 up in weight, then rest-pause to failure",
        "Close Grip Pulldown + Pullover — 3×12 superset, no rest",
        "Rear Delt Fly (lying) — 3×20",
        "Back Hyperextensions — 3×30",
        "Abs: 80 crunches w/ pause + lying crunches"]},
    1: {"focus": "Chest, Shoulders & Biceps", "kind": "lift", "exercises": [
        "Incline Barbell Press — 15 warm-up, 12, 2×8 + drop set",
        "Incline Dumbbell Fly — 3×12",
        "Flat Dumbbell Press — 3×12, heavier each set",
        "Cable Crossover — 3×20",
        "Parallel Bar Dips — 3×15",
        "Lateral Raises — 3×20",
        "Preacher Curl — 20 light, 15, 12, 8 heavy",
        "Standing Barbell Curl — 3×12 down to 8",
        "Seated Dumbbell Curl — 3×12",
        "Abs: 80 crunches w/ pause + lying crunches"]},
    2: {"focus": "Cardio — Zone 2", "kind": "cardio", "exercises": CARDIO},
    3: {"focus": "Legs — Quads, Hams, Glutes", "kind": "lift", "exercises": [
        "Leg Extension — 3×20 (quad warm-up)",
        "Leg Press — 4×15 down to 12, heavier each set",
        "Barbell Squats — 3×12, heavier each set",
        "Lying Leg Curl — 3×12",
        "Walking Lunges (weighted) — 3×20 steps",
        "Hyperextensions — 3×20",
        "Abs: 80 crunches w/ pause + lying crunches"]},
    4: {"focus": "Shoulders & Triceps", "kind": "lift", "exercises": [
        "Barbell Shrugs — 3×12 down to 8, heavier each set",
        "Seated Shoulder Press Machine — 3×12 down to 8",
        "Seated Lateral Raises — 3×20 down to 12",
        "Front Raises — 3×20 down to 12",
        "Rear Delt Fly (lying) — 3×20",
        "Cable Shoulder Giant Set — lateral → front (rope) → rear delt fly, 3 rounds no rest",
        "Cable Tricep Pushdown — 4×20 down to 12",
        "Skull Crushers — 3×15 down to 8",
        "Single-Arm Cable Tricep Ext — 3×15 each arm",
        "Abs: 80 crunches w/ pause + lying crunches"]},
    5: {"focus": "Cardio — Zone 2", "kind": "cardio", "exercises": CARDIO},
    6: {"focus": "Full Rest", "kind": "rest", "exercises": [
        "No workout, no run, no exceptions. Recovery is the work today."]},
}

SUPPLEMENTS = [
    {"name": "Creatine monohydrate", "dose": "5g", "when": "morning"},
    {"name": "Vitamin D3 + K2",      "dose": "",   "when": "morning"},
    {"name": "Fish oil (omega-3)",   "dose": "",   "when": "with a meal"},
    {"name": "Magnesium glycinate",  "dose": "300–400mg", "when": "before bed"},
]

NON_NEGOTIABLES = [
    "Hydration: 3–4 L water every day",
    "Sleep: 7–9 hours",
    "Weigh in every morning, same time, after bathroom — judge only the 7-day average",
    "Log everything (MyFitnessPal/Cronometer) the first 4 weeks",
]


def _d(on: Optional[date]) -> date:
    return on or datetime.now().date()


def todays_focus(on: Optional[date] = None) -> Dict:
    """The split entry for a given date (focus label, kind, exercises)."""
    return SPLIT[_d(on).weekday()]


def workout_for_weekday(weekday: int) -> Dict:
    return SPLIT[weekday % 7]


def plan_day_number(on: Optional[date] = None) -> int:
    """1-based day count since the reset (negative/zero before it starts)."""
    return (_d(on) - RESET_DATE).days + 1


def phase_for(on: Optional[date] = None) -> Optional[Dict]:
    d = _d(on)
    for p in PHASES:
        if p["start"] <= d and (p["end"] is None or d <= p["end"]):
            return p
    return None


def is_refeed_day(on: Optional[date] = None) -> bool:
    d = _d(on)
    if d < RESET_DATE:
        return False
    return (d - RESET_DATE).days % REFEED_EVERY_DAYS == 0 and (d - RESET_DATE).days > 0


def next_refeed(on: Optional[date] = None) -> date:
    d = _d(on)
    delta = (d - RESET_DATE).days
    if delta < 0:
        return RESET_DATE + timedelta(days=REFEED_EVERY_DAYS)
    k = (delta // REFEED_EVERY_DAYS) + 1
    return RESET_DATE + timedelta(days=k * REFEED_EVERY_DAYS)


def memory_summary() -> List[str]:
    """One-liners stored in Jarvis's persistent memory so he always knows the
    plan without re-reading the file."""
    return [
        f"Geo is on a Summer Reset weight-loss plan: {START_WEIGHT:.0f} lbs → "
        f"{GOAL_WEIGHT:.0f} lbs goal, started {RESET_DATE:%b %-d %Y}.",
        f"Geo's daily macro targets are {MACROS['calories']} cal, "
        f"{MACROS['protein']}g protein, {MACROS['carbs']}g carbs, {MACROS['fat']}g fat "
        f"(refeed to {REFEED_CALORIES} cal every 2 weeks).",
        "Geo's training split: Mon Back, Tue Chest/Biceps, Wed cardio, Thu Legs, "
        "Fri Shoulders/Triceps, Sat cardio, Sun full rest.",
        "Geo weighs in every morning and judges progress on the 7-day average, not single days.",
    ]
