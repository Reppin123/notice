"""What the eyes were on during an utterance -> one short note for the agent (low tokens, no image).

Without word timings (phone call turns), take fixations in [start - 1 s, end + 0.3 s] and keep up to
three distinct targets in time order; the model matches them to "this" / "there" from the words.
"""
from __future__ import annotations

import difflib
import time

import cv2

from . import screen
from .core import AttentionState, Fixation
from .scene import colour_hist, same_colours

LEAD_S, TAIL_S = 1.0, 0.3
MIN_S = 0.15
STALE_S = 1.5  # no gaze for this long = glasses not streaming


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
    # keep the 5 longest, in time order (build() merges looks at one object, then caps at 3)
    keep = sorted(sorted(merged, key=lambda f: -f.duration)[:5], key=lambda f: f.start)
    return keep


VISUAL = {"colour", "color", "look", "looks", "picture", "photo", "image", "shape", "wearing", "logo",
          "design", "style", "bottle", "brand", "see", "pretty", "ugly", "size"}
THIN_TEXT = 15  # fewer characters than this = the picture is what matters


def build(state: AttentionState, memory, start: float, end: float, words: str = "") -> dict:
    t0 = time.time()
    g = state.last_gaze()
    if g is None or t0 - g.t > STALE_S:
        age = "never" if g is None else f"{t0 - g.t:.0f} s ago"
        return {"items": [], "stale": True, "build_ms": 0,
                "note": f"[Glasses: NOT streaming right now (last gaze {age}). You can't see what the user is "
                        "looking at; if they ask about something in front of them, say so.]"}
    items, seen_rows = [], set()
    for f in targets(state, start, end):
        it = {"t": round(f.start - start, 2), "dwell_ms": round(f.duration * 1000)}
        if f.x is not None:
            c = screen.candidates_near(f.x, f.y, max_n=3)
            it.update(kind="screen", x=round(f.x), y=round(f.y), elements=[
                {"role": e["role"].replace("AX", ""), "label": e["label"][:80], "app": e["app"], "center": e["center"]} for e in c])
        elif memory is not None and f.scene_x is not None:
            rid = memory.on_fixation(f, force=True)  # the user spoke about it: keep it even if short
            row = memory.get(rid) if rid else None
            if row is None or row["id"] in seen_rows:
                continue  # not armed, a face, or the same thing already listed
            seen_rows.add(row["id"])
            it.update(kind="world", rows=[row], sx=f.scene_x, sy=f.scene_y, tm=(f.start + f.end) / 2)
        else:
            continue
        items.append(it)
    visual = bool(VISUAL & {w.strip(".,?!").lower() for w in words.split()})
    items = _merge_same_object(items)
    items = sorted(sorted(items, key=lambda i: -i["dwell_ms"])[:3], key=lambda i: i["t"])
    items = [_finish(it, visual) if it["kind"] == "world" else it for it in items]
    return {"items": items, "note": note(items), "build_ms": round((time.time() - t0) * 1000)}


SAME_COLOURS = 0.5     # real Neon crops of one held bag: 0.52-0.94; vs most of the room <= 0.26
SAME_COLOURS_NO_TEXT = 0.8  # without text to confirm, colours alone mislead: phone on a wood desk next to
                            # the bag scored 0.62 against it
TEXT_AGREES = 0.35     # garbled reads of one logo: >= 0.38. Unrelated labels usually score lower, but ones sharing
                       # a word can reach 0.5 ("Trader Joes milk" / "Oat Milk"); colours must also match then


SURE_OCR = 0.8  # Vision line confidence: clean print reads 1.0, garbled logos 0.5-0.75


def _junk(t: str) -> bool:
    """Mixed scripts or mostly non-letters = a garbled read."""
    letters = [c for c in t if c.isalpha()]
    return bool(letters) and (sum(not c.isascii() for c in letters) / len(letters) > 0.1 or len(letters) < len(t) * 0.4)


def _norm(t: str) -> str:
    return "".join(c for c in t.lower() if c.isalnum() or c == " ").strip()


def _same_object(a: dict, b: dict) -> bool:
    ha, hb = _hist(a), _hist(b)
    if ha is None or hb is None:
        return False
    colours = same_colours(ha, hb)
    ta, tb = _norm(a.get("caption") or ""), _norm(b.get("caption") or "")
    if ta and tb:
        return colours >= SAME_COLOURS and difflib.SequenceMatcher(None, ta, tb).ratio() >= TEXT_AGREES
    return colours >= SAME_COLOURS_NO_TEXT


def _merge_same_object(items: list[dict]) -> list[dict]:
    """Back-to-back looks at one object (eyes wander over a bag you hold up) become ONE item.
    Scene position can't decide this (the object and head move); colours + not-clashing text can."""
    out: list[dict] = []
    for it in items:
        p = out[-1] if out else None
        if p and p["kind"] == it["kind"] == "world":
            if _same_object(p["rows"][-1], it["rows"][0]):
                p["rows"] += it["rows"]
                p["dwell_ms"] += it["dwell_ms"]
                continue
        out.append(it)
    return out


def _hist(row: dict):
    if "_hist" not in row:
        img = cv2.imread(row["crop"]) if row.get("crop") else None
        row["_hist"] = colour_hist(img) if img is not None else None
    return row["_hist"]


def _finish(it: dict, visual: bool) -> dict:
    """Keep the reading OCR was surest of; send the picture when the text is thin, the words are about
    looks, or the reads disagree (a stylised logo: let the model read it)."""
    rows = it.pop("rows")
    best = max(rows, key=lambda r: (r.get("conf") or 0, len(r.get("caption") or "")))
    text = best.get("caption") or ""
    reads = {_norm(r.get("caption") or "") for r in rows if r.get("caption")}
    disagree = len(reads) > 1
    codes = " ".join(dict.fromkeys(c for r in rows for c in (r.get("codes") or "").split()))
    labels = best.get("labels") or next((r["labels"] for r in rows if r.get("labels")), "")
    thin = len(text) < THIN_TEXT and not codes
    unsure = (best.get("conf") or 0) < SURE_OCR or _junk(text)
    if text and (unsure or disagree):
        labels = ""  # Vision's labels on a held object ("machine") mislead more than help next to text
    it.update(caption=text, codes=codes, labels=labels, looks=len(rows), unsure=bool(text) and (unsure or disagree),
              image=best["crop"] if (thin or visual or disagree or unsure) else None)
    return it


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
            cap = it["caption"]
            if cap and it.get("unsure"):
                txt = f'something whose text OCR read unsure as "{cap}" (trust the picture over this)'
            elif cap:
                txt = f'something with the text "{cap}"'
            else:
                txt = "something with no readable text"
            if it.get("labels"):
                txt += f' (looks like: {it["labels"]})'
            if it.get("codes"):
                txt += f' (QR/barcode: {it["codes"]})'
            img = f" [see it: look_through_glasses item={i}]" if it.get("image") else ""
            parts.append(f"{i}) in the room: {txt}{img}")
    return ("[Glasses: while speaking the user looked at, in order (" + str(len(items)) +
            "); 'this/that/there' refer to these] " + " · ".join(parts))
