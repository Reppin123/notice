"""Snap a noisy screen fixation to nearby accessibility elements (macOS AX API).

Gaze is ~1.5 deg accurate (~60 px), so we never trust the pixel: sample rings around the
fixation, collect distinct elements, and climb to a meaningful ancestor (row, button, link...).
"""
from __future__ import annotations

import math

from AppKit import NSWorkspace
from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementCopyElementAtPosition,
    AXUIElementCreateSystemWide,
    AXUIElementGetPid,
    AXValueGetValue,
    kAXValueCGPointType,
    kAXValueCGSizeType,
)

_SYSTEM = AXUIElementCreateSystemWide()

# Roles a person would plausibly mean by "this". Leaf text gets lifted to these.
MEANINGFUL = {
    "AXRow", "AXCell", "AXButton", "AXLink", "AXMenuItem", "AXCheckBox", "AXRadioButton",
    "AXTextField", "AXTextArea", "AXImage", "AXPopUpButton", "AXTab", "AXHeading", "AXDisclosureTriangle",
}
TEXTY = {"AXStaticText", "AXTextArea", "AXTextField", "AXHeading"}


def _attr(el, name):
    err, v = AXUIElementCopyAttributeValue(el, name, None)
    return v if err == 0 else None


def _frame(el):
    pos, size = _attr(el, "AXPosition"), _attr(el, "AXSize")
    if pos is None or size is None:
        return None
    _, p = AXValueGetValue(pos, kAXValueCGPointType, None)
    _, s = AXValueGetValue(size, kAXValueCGSizeType, None)
    return (p.x, p.y, s.width, s.height)


def _label(el) -> str:
    for a in ("AXTitle", "AXDescription", "AXValue", "AXHelp"):
        v = _attr(el, a)
        if isinstance(v, str) and v.strip():
            return v.strip()[:200]
    # rows/cells: gather child text
    kids = _attr(el, "AXChildren") or []
    parts = []
    for k in list(kids)[:8]:
        t = _attr(k, "AXValue") or _attr(k, "AXTitle") or _attr(k, "AXDescription")
        if isinstance(t, str) and t.strip():
            parts.append(t.strip())
        else:
            for kk in list(_attr(k, "AXChildren") or [])[:6]:
                t = _attr(kk, "AXValue") or _attr(kk, "AXTitle")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
    return " · ".join(parts)[:240]


def element_at(x: float, y: float):
    err, el = AXUIElementCopyElementAtPosition(_SYSTEM, float(x), float(y), None)
    return el if err == 0 else None


def _lift(el, max_up: int = 4):
    """Return the leaf plus the nearest meaningful ancestor (may be the leaf itself)."""
    cur, best = el, None
    for _ in range(max_up + 1):
        if cur is None:
            break
        if _attr(cur, "AXRole") in MEANINGFUL:
            best = cur
            break
        cur = _attr(cur, "AXParent")
    return best or el


def _app_name(el) -> str:
    err, pid = AXUIElementGetPid(el, None)
    if err != 0:
        return ""
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if app.processIdentifier() == pid:
            return app.localizedName() or ""
    return ""


def describe(el, fx: float, fy: float) -> dict | None:
    fr = _frame(el)
    if fr is None:
        return None
    x, y, w, h = fr
    # distance from fixation to the element's box (0 if inside)
    dx = max(x - fx, 0, fx - (x + w))
    dy = max(y - fy, 0, fy - (y + h))
    return {
        "role": _attr(el, "AXRole") or "",
        "subrole": _attr(el, "AXSubrole") or "",
        "label": _label(el),
        "frame": [round(x), round(y), round(w), round(h)],
        "center": [round(x + w / 2), round(y + h / 2)],
        "distance_px": round(math.hypot(dx, dy)),
        "app": _app_name(el),
    }


def candidates_near(fx: float, fy: float, radius: float = 70, max_n: int = 6) -> list[dict]:
    seen, out = set(), []
    pts = [(fx, fy)]
    for r in (radius / 3, 2 * radius / 3, radius):
        for k in range(8):
            a = k * math.pi / 4
            pts.append((fx + r * math.cos(a), fy + r * math.sin(a)))
    for px, py in pts:
        leaf = element_at(px, py)
        if leaf is None:
            continue
        for el in {id(leaf): leaf, id(m := _lift(leaf)): m}.values():
            d = describe(el, fx, fy)
            if not d or d["frame"][2] * d["frame"][3] > 1200 * 900:  # skip whole windows/web areas
                continue
            key = (d["role"], d["label"], tuple(d["frame"]))
            if key in seen:
                continue
            seen.add(key)
            d["meaningful"] = d["role"] in MEANINGFUL
            out.append(d)
    # Closest first; meaningful elements win ties; tiny boxes are penalised a bit (likely icons).
    out.sort(key=lambda d: (d["distance_px"] - (15 if d["meaningful"] else 0), -d["frame"][2] * d["frame"][3] / 1e5))
    return out[:max_n]
