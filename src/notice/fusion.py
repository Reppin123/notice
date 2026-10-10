"""Align spoken words with fixations. People look at a thing ~0.3-1 s BEFORE they name it."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .core import AttentionState, Fixation

DEICTICS = {"this", "that", "these", "those", "here", "there", "it", "him", "her", "them"}
STRONG = {"this", "that", "these", "those", "here", "there"}  # "it" only counts when nothing strong is said

BEFORE_S = 0.9
AFTER_S = 0.2
MIN_FIX_S = 0.15


@dataclass
class Word:
    text: str
    start: float  # Mac unix seconds
    end: float

    @property
    def norm(self) -> str:
        return re.sub(r"[^a-z']", "", self.text.lower())


def pick_fixation(state: AttentionState, t: float, exclude: set[int] = frozenset()) -> Fixation | None:
    """Latest fixation >= 150 ms overlapping [t-900ms, t+200ms]; fall back to the longest short one."""
    fx = [f for f in state.fixations_between(t - BEFORE_S, t + AFTER_S) if id(f) not in exclude]
    long_enough = [f for f in fx if f.duration >= MIN_FIX_S]
    if long_enough:
        return max(long_enough, key=lambda f: f.start)
    return max(fx, key=lambda f: f.duration) if fx else None


def deictic_words(words: list[Word]) -> list[Word]:
    strong = [w for w in words if w.norm in STRONG]
    return strong or [w for w in words if w.norm in DEICTICS]


def resolve(state: AttentionState, words: list[Word], snap=None) -> list[dict]:
    """For each deictic word: the fixation it points to and (screen mode) nearby elements.

    "move this there" uses two different fixations in order; a fixation used by an earlier
    word is excluded so the second word can't collapse onto the first target.
    """
    used: set[int] = set()
    out = []
    for w in deictic_words(words):
        f = pick_fixation(state, w.start, used)
        item = {"word": w.text, "t": w.start, "fixation": f.to_dict() if f else None, "candidates": []}
        if f:
            used.add(id(f))
            if f.on_screen and snap:
                item["candidates"] = snap(f.x, f.y)
        out.append(item)
    return out
