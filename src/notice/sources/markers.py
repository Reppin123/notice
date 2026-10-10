"""On-screen AprilTag layout (tag36h11 ids 0-3 at the four screen corners), in screen points."""
from __future__ import annotations

import os

SIZE = int(os.environ.get("NOTICE_TAG_PT", 140))  # black-square side in points; bigger = easier for the wide FOV
PAD = int(os.environ.get("NOTICE_TAG_PAD", 24))  # white quiet zone around each tag


def screen_size() -> tuple[float, float]:
    from AppKit import NSScreen
    f = NSScreen.mainScreen().frame()
    return f.size.width, f.size.height


def tag_origins() -> dict[int, tuple[float, float]]:
    """Top-left of each tag's black square, screen points, top-left origin."""
    w, h = screen_size()
    o = PAD
    return {0: (o, o), 1: (w - o - SIZE, o), 2: (w - o - SIZE, h - o - SIZE), 3: (o, h - o - SIZE)}


def marker_verts():
    verts = {}
    for i, (x, y) in tag_origins().items():
        verts[i] = [(x, y), (x + SIZE, y), (x + SIZE, y + SIZE), (x, y + SIZE)]  # TL, TR, BR, BL
    return verts, screen_size()


def marker_png(marker_id: int, path: str, side: int = 512):
    import cv2
    from pupil_labs.real_time_screen_gaze.marker_generator import generate_marker
    img = generate_marker(marker_id, side_pixels=8)  # black border + 6x6 data cells
    img = cv2.resize(img, (side, side), interpolation=cv2.INTER_NEAREST)
    q = side // 8  # white quiet zone (one cell) so the detector can find the edge
    img = cv2.copyMakeBorder(img, q, q, q, q, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, img)
