"""Smart zoom: gaze is a hint, not the crop centre. On real calls Neon's gaze sat ~100-150 px above held
objects at reading distance; the zoom must still land on the thing meant (the text block near the gaze)."""
from pathlib import Path

import cv2

from notice import snapshot as snap

WALL = cv2.imread(str(Path(__file__).parent.parent / "testdata" / "poster_wall.jpg"))
H, W = WALL.shape[:2]


def box(area):
    x, y, w, h = (float(v) for v in area.split(","))
    return x * W, y * H, (x + w) * W, (y + h) * H


def contains(area, poster):
    x0, y0, x1, y1 = box(area)
    px0, py0, px1, py1 = poster
    return x0 <= px0 and y0 <= py0 and x1 >= px1 and y1 >= py1


PUPIL_TEXT = (86, 159, 347, 474)        # the four text lines of the PUPIL LABS poster
ELEVEN_TEXT = (596, 159, 885, 474)


def test_text_lines_group_per_poster_not_across_posters():
    r = __import__("notice.ocr", fromlist=["regions"]).regions(WALL)
    px = [(int(b[0] * W), int(b[1] * H), int((b[0] + b[2]) * W), int((b[1] + b[3]) * H)) for b in r["text"]]
    blocks = snap._merge_text(px)
    assert len(blocks) == 6


def test_gaze_on_the_text_zooms_to_its_block():
    _, m = snap.zoom(WALL, 200, 270)
    assert m["how"] == "text" and contains(m["area"], PUPIL_TEXT)


def test_gaze_120px_above_still_zooms_to_the_poster():
    _, m = snap.zoom(WALL, 200, 40)
    assert m["how"] == "text" and contains(m["area"], PUPIL_TEXT)


def test_gaze_in_the_gap_between_posters_picks_the_nearer_one():
    _, m = snap.zoom(WALL, 545, 300)
    assert contains(m["area"], ELEVEN_TEXT)


def test_model_chosen_area_wins_over_gaze():
    out, m = snap.zoom(WALL, 200, 270, area="0.69,0.6,0.27,0.3")  # the YOGA poster
    assert m["how"] == "area" and out.shape[1] >= 400


def test_nothing_near_falls_back_to_a_gaze_crop():
    _, m = snap.zoom(WALL, 1590, 1190)
    assert m["how"] in ("gaze", "object")


def test_whole_and_sheet_are_one_1280_image():
    assert snap.whole(WALL, 200, 270).shape[:2] == (960, 1280)
    assert snap.sheet([(WALL, 200, 270, "1"), (WALL, 700, 270, "2")]).shape[:2] == (960, 1280)
