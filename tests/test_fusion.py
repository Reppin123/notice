from notice.core import AttentionState, Fixation, GazePoint, IDTDetector
from notice.fusion import Word, resolve


def st(*fx):
    s = AttentionState()
    for f in fx:
        s.add_fixation(f)
    return s


def test_looks_before_speaking():
    # looked at A, then at B 0.6 s before saying "this" -> B
    s = st(Fixation(9.0, 9.3, 100, 100), Fixation(9.4, 9.9, 500, 500))
    r = resolve(s, [Word("reply", 10.0, 10.2), Word("to", 10.2, 10.3), Word("this", 10.3, 10.5)])
    assert r[0]["fixation"]["x"] == 500


def test_move_this_there_uses_two_fixations():
    s = st(Fixation(9.0, 9.6, 100, 100), Fixation(10.0, 10.5, 800, 300))
    words = [Word("move", 9.4, 9.6), Word("this", 9.6, 9.8), Word("there", 10.6, 10.8)]
    r = resolve(s, words)
    assert [x["fixation"]["x"] for x in r] == [100, 800]


def test_ignores_short_glances_when_a_real_fixation_exists():
    s = st(Fixation(9.2, 9.6, 300, 300), Fixation(9.7, 9.78, 900, 900))
    r = resolve(s, [Word("that", 9.8, 10.0)])
    assert r[0]["fixation"]["x"] == 300


def test_it_only_when_no_strong_deictic():
    s = st(Fixation(9.2, 9.6, 300, 300))
    assert [x["word"] for x in resolve(s, [Word("open", 9.7, 9.8), Word("it", 9.8, 9.9)])] == ["it"]
    assert [x["word"] for x in resolve(s, [Word("put", 9.6, 9.7), Word("it", 9.7, 9.8), Word("there", 9.8, 9.9)])] == ["there"]


def test_idt_detector():
    out = []
    d = IDTDetector(out.append)
    for i in range(30):
        d.feed(GazePoint(i / 100, 100 + (i % 3), 100))
    for i in range(30, 60):
        d.feed(GazePoint(i / 100, 600, 400))
    d.feed(GazePoint(0.61, None, None))
    assert len(out) == 2 and abs(out[0].x - 101) < 2 and out[1].x == 600
