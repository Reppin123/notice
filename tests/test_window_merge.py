"""Same-object merging on real Neon crops (2026-10-10): eyes wandering over a held Cheetos bag gave three
looks with three OCR reads; they must become one item with the confident read. The phone beside the bag
(similar wood/orange colours, no text) must stay separate."""
from pathlib import Path

from notice import window

F = Path(__file__).parent / "fixtures" / "neon"


def item(t, name, caption, conf):
    return {"t": t, "dwell_ms": 800, "kind": "world",
            "rows": [{"id": t, "crop": str(F / f"{name}.jpg"), "caption": caption, "conf": conf, "codes": "", "labels": ""}]}


def run(items):
    return [window._finish(it, False) for it in window._merge_same_object(items)]


def bag():
    return [item(0, "bag_1", "Cheetos Crunchy", 1.0), item(1, "bag_2", "freetos (1B0)", 0.5),
            item(2, "bag_3", "geetas 1507", 0.75)]


def test_three_looks_at_one_bag_become_one_item_with_the_confident_read():
    out = run(bag())
    assert [(o["looks"], o["caption"]) for o in out] == [(3, "Cheetos Crunchy")]
    assert out[0]["image"]  # reads disagreed: the model gets the picture to read the logo itself


def test_phone_next_to_the_bag_stays_separate():
    out = run(bag() + [item(3, "phone_by_bag", "", 0.0)])
    assert [o["looks"] for o in out] == [3, 1]


def test_unrelated_text_is_not_merged_even_with_matching_colours():
    out = run([item(0, "bag_1", "Cheetos Crunchy", 1.0), item(1, "bag_2", "Trader Joes milk", 1.0)])
    assert len(out) == 2
