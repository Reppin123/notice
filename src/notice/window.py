"""What the eyes were on during an utterance -> one short note for the agent (low tokens, no image).

Without word timings (phone call turns), take fixations in [start - 1 s, end + 0.3 s] and keep up to
three distinct targets in time order; the model matches them to "this" / "there" from the words.
"""
from __future__ import annotations

import time

from . import screen
from .core import AttentionState, Fixation

LEAD_S, TAIL_S = 1.0, 0.3
MIN_S = 0.15


def _current_world_fixation(state: AttentionState, end: float) -> Fixation | None:
    """Neon reports a fixation when it ENDS; if the user is still looking, build one from raw gaze."""
    with state.lock:
        pts = [g for g in state.gaze if g.t >= end - 0.5 and g.sx is not None]
    if len(pts) < 5:
        return None
    xs, ys = [p.sx for p in pts], [p.sy for p in pts]
    if (max(xs) - min(xs)) + (max(ys) - min(ys)) > 80:  # scene px; not stable
        return None
    sx, sy = sum(xs) / len(xs), sum(ys) / len(ys)
    x = y = None
    if all(p.x is not None for p in pts):
        x, y = sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts)
    return Fixation(pts[0].t, pts[-1].t, x, y, sx, sy, source="live")


def targets(state: AttentionState, start: float, end: float) -> list[Fixation]:
    fx = [f for f in state.fixations_between(start - LEAD_S, end + TAIL_S) if f.duration >= MIN_S]
    cur = _current_world_fixation(state, end + TAIL_S)
    if cur and not any(abs(f.end - cur.end) < 0.2 for f in fx):
        fx.append(cur)
    fx.sort(key=lambda f: f.start)
    # merge consecutive fixations on the same spot (re-fixations on one poster)
    merged: list[Fixation] = []
    for f in fx:
        p = merged[-1] if merged else None
        same = p and ((f.x is not None and p.x is not None and abs(f.x - p.x) + abs(f.y - p.y) < 80) or
                      (f.x is None and p.x is None and f.scene_x is not None and p.scene_x is not None and
                       abs(f.scene_x - p.scene_x) + abs(f.scene_y - p.scene_y) < 120))
        if same:
            p.end = max(p.end, f.end)
        else:
            merged.append(Fixation(f.start, f.end, f.x, f.y, f.scene_x, f.scene_y, f.source))
    # keep the 3 longest, in time order
    keep = sorted(sorted(merged, key=lambda f: -f.duration)[:3], key=lambda f: f.start)
    return keep


def build(state: AttentionState, memory, start: float, end: float, caption_wait_s: float = 1.2) -> dict:
    t0 = time.time()
    items = []
    for f in targets(state, start, end):
        it = {"t": round(f.start - start, 2), "dwell_ms": round(f.duration * 1000)}
        if f.x is not None:
            c = screen.candidates_near(f.x, f.y, max_n=3)
            it.update(kind="screen", x=round(f.x), y=round(f.y), elements=[
                {"role": e["role"].replace("AX", ""), "label": e["label"][:80], "app": e["app"], "center": e["center"]} for e in c])
        elif memory is not None and f.scene_x is not None:
            f2 = Fixation(f.start, max(f.end, f.start + 0.3), None, None, f.scene_x, f.scene_y)  # store even if short
            rid = memory.on_fixation(f2)
            row = memory.get(rid, wait_s=caption_wait_s) if rid else None
            it.update(kind="world", caption=(row or {}).get("caption", ""), image=(row or {}).get("crop"))
        else:
            continue
        items.append(it)
    return {"items": items, "note": note(items), "build_ms": round((time.time() - t0) * 1000)}


def note(items: list[dict]) -> str:
    if not items:
        return ""
    parts = []
    for i, it in enumerate(items, 1):
        if it["kind"] == "screen":
            e = it["elements"][0] if it["elements"] else None
            what = f'{e["role"]} "{e["label"]}" in {e["app"]} at {tuple(e["center"])}' if e else f'screen point ({it["x"]},{it["y"]})'
            parts.append(f"{i}) on the Mac screen: {what}")
        else:
            cap = it["caption"] or "(not captioned yet; open the image)"
            parts.append(f"{i}) in the room: {cap} [image: {it['image']}]")
    return ("[Glasses: while speaking the user looked at, in order (" + str(len(items)) +
            "); 'this/that/there' refer to these] " + " · ".join(parts))
