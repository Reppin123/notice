"""What the model sees when it looks through the glasses: one image per look.

- whole:  the full scene at 1280 px (80 % of native: most small print survives) with the gaze circle.
          The default. It tolerates gaze error (at reading distance Neon's gaze sat above held objects).
- sheet:  2x2 grid of the frames of the things looked at while speaking, numbered like the note, so
          one look covers "which one" instead of three tool calls. Tiles are 1/3 native: not for reading.
- motion: 2x2 grid of 4 frames across the utterance, for "what just happened".
- zoom:   native-resolution crop. With `area` the model picks the region from the whole view it saw;
          without, it snaps to the object or text block nearest the gaze (Apple Vision), so a gaze dot
          that lands just above a page still zooms to the page. Plain gaze crop only as a last resort.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import ocr
from .scene import mark_gaze

WHOLE_W = 1280
TILE = (640, 480)
NEAR = 0.12        # a candidate within 12 % of frame width of the gaze still counts (offset seen: ~100-150 px)
MIN_ZOOM = 400     # px; smaller boxes grow around their centre (single words need context)
MARGIN = 0.08


def whole(img: np.ndarray, sx: float, sy: float) -> np.ndarray:
    return mark_gaze(img, sx, sy, max_w=WHOLE_W)


def sheet(frames: list[tuple[np.ndarray, float, float, str]]) -> np.ndarray:
    """Up to 4 (image, gaze x, gaze y, label) in a 2x2 grid of 640x480 tiles."""
    tiles = []
    for img, sx, sy, label in frames[:4]:
        h, w = img.shape[:2]
        t = cv2.resize(img, TILE, interpolation=cv2.INTER_AREA)
        gx, gy = int(sx * TILE[0] / w), int(sy * TILE[1] / h)
        cv2.circle(t, (gx, gy), 16, (0, 0, 255), 3)
        cv2.rectangle(t, (0, 0), (64, 52), (0, 0, 0), -1)
        cv2.putText(t, label, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        tiles.append(t)
    while len(tiles) < 4:
        tiles.append(np.full((TILE[1], TILE[0], 3), 40, np.uint8))
    rows = [np.hstack(tiles[:2]), np.hstack(tiles[2:])]
    grid = np.vstack(rows)
    cv2.line(grid, (TILE[0], 0), (TILE[0], grid.shape[0]), (255, 255, 255), 2)
    cv2.line(grid, (0, TILE[1]), (grid.shape[1], TILE[1]), (255, 255, 255), 2)
    return grid


def _merge_text(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Text lines -> blocks the way a page is laid out. Vision sometimes returns one box spanning
    many lines (on the test wall: half the image), so anything taller than 3 typical lines is dropped.
    Lines join when they share a column (overlap horizontally) and sit within ~3.5 line heights, or
    are words on one row; side-by-side posters stay separate."""
    if not boxes:
        return []
    hs = sorted(b[3] - b[1] for b in boxes)
    line_h = hs[len(hs) // 2]
    # [x0, y0, x1, y1, line height]: a block keeps its LINE height, never its own (growing) height
    blocks = [list(b) + [b[3] - b[1]] for b in boxes if b[3] - b[1] <= 3 * line_h]

    def joins(a, b):
        lh = max(a[4], b[4])  # posters space lines ~3 heights apart; documents ~1.3
        overlap = min(a[2], b[2]) - max(a[0], b[0])
        vgap = max(a[1], b[1]) - min(a[3], b[3])
        hgap = max(a[0], b[0]) - min(a[2], b[2])
        same_column = overlap > 0.3 * min(a[2] - a[0], b[2] - b[0]) and vgap <= 3.5 * lh
        same_row = vgap < 0 and hgap <= 1.2 * lh
        return same_column or same_row

    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                if joins(blocks[i], blocks[j]):
                    a, b = blocks[i], blocks[j]
                    blocks[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), max(a[4], b[4])]
                    del blocks[j]
                    changed = True
                    break
            if changed:
                break
    return [tuple(b[:4]) for b in blocks]


def _dist(box, x, y) -> float:
    dx = max(box[0] - x, 0, x - box[2])
    dy = max(box[1] - y, 0, y - box[3])
    return (dx * dx + dy * dy) ** 0.5


def zoom_box(img: np.ndarray, sx: float, sy: float, area: str | None = None) -> tuple[tuple[int, int, int, int], str]:
    """(x0, y0, x1, y1) in frame px and how it was chosen: 'area' | 'text' | 'object' | 'gaze'."""
    h, w = img.shape[:2]
    if area:
        fx, fy, fw, fh = (float(v) for v in area.split(","))
        box, how = (int(fx * w), int(fy * h), int((fx + fw) * w), int((fy + fh) * h)), "area"
    else:
        regions = ocr.regions(img)
        to_px = lambda b: (int(b[0] * w), int(b[1] * h), int((b[0] + b[2]) * w), int((b[1] + b[3]) * h))
        cands = [(b, "text") for b in _merge_text([to_px(b) for b in regions["text"]])]
        cands += [(to_px(b), "object") for b in regions["objects"]
                  if b[2] * b[3] < 0.6]  # "the whole scene" is not a zoom
        near = [(c, how) for c, how in cands if _dist(c, sx, sy) <= NEAR * w]
        if near:
            # text near the gaze wins (reading is the common zoom; Vision's object boxes are rough and
            # on the test wall spanned two posters); then the nearest, then the tighter box
            box, how = min(near, key=lambda ch: (ch[1] != "text", round(_dist(ch[0], sx, sy) / (0.02 * w)),
                                                 (ch[0][2] - ch[0][0]) * (ch[0][3] - ch[0][1])))
        else:
            box, how = (int(sx - 320), int(sy - 320), int(sx + 320), int(sy + 320)), "gaze"
    x0, y0, x1, y1 = box
    mx, my = int((x1 - x0) * MARGIN), int((y1 - y0) * MARGIN)
    x0, y0, x1, y1 = x0 - mx, y0 - my, x1 + mx, y1 + my
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max(x1 - x0, MIN_ZOOM), max(y1 - y0, MIN_ZOOM)
    x0, y0 = int(min(max(cx - bw / 2, 0), max(w - bw, 0))), int(min(max(cy - bh / 2, 0), max(h - bh, 0)))
    return (x0, y0, int(min(x0 + bw, w)), int(min(y0 + bh, h))), how


def zoom(img: np.ndarray, sx: float, sy: float, area: str | None = None) -> tuple[np.ndarray, dict]:
    (x0, y0, x1, y1), how = zoom_box(img, sx, sy, area)
    out = img[y0:y1, x0:x1].copy()
    if x0 <= sx < x1 and y0 <= sy < y1:
        cv2.circle(out, (int(sx - x0), int(sy - y0)), 18, (0, 0, 255), 2)
    if max(out.shape[:2]) > WHOLE_W:
        s = WHOLE_W / max(out.shape[:2])
        out = cv2.resize(out, (int(out.shape[1] * s), int(out.shape[0] * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    return out, {"how": how, "area": f"{x0 / w:.2f},{y0 / h:.2f},{(x1 - x0) / w:.2f},{(y1 - y0) / h:.2f}"}
