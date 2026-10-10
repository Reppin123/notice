"""Mouse-as-gaze simulator with Neon-like error: a slowly drifting offset (accuracy ~1.5 deg,
up to ~60 px) plus small jitter, so element snapping gets exercised.

Looking away: park the cursor in the top-left corner (x<5, y<5) or hold Fn... corner is enough for dev.
"""
from __future__ import annotations

import random
import threading
import time

import Quartz

from ..core import AttentionState, GazePoint, IDTDetector


def mouse_pos() -> tuple[float, float]:
    p = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return p.x, p.y


class MouseGaze:
    def __init__(self, state: AttentionState, hz: float = 120, offset_px: float = 35.0, jitter_px: float = 2.0, recorder=None):
        self.state = state
        self.hz = hz
        self.offset_px, self.jitter_px = offset_px, jitter_px
        self.off = [0.0, 0.0]
        self.recorder = recorder
        self.detector = IDTDetector(state.add_fixation)
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="mouse-gaze").start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        period = 1.0 / self.hz
        while not self._stop.is_set():
            t = time.time()
            x, y = mouse_pos()
            if x < 5 and y < 5:  # "looking away"
                g = GazePoint(t, None, None)
            else:
                for i in range(2):  # mean-reverting random walk, sd ~ offset_px
                    self.off[i] += -0.0005 * self.off[i] + random.gauss(0, self.offset_px * 0.032)
                j = self.jitter_px
                g = GazePoint(t, x + self.off[0] + random.gauss(0, j), y + self.off[1] + random.gauss(0, j))
            self.state.add_gaze(g)
            self.detector.feed(g)
            if self.recorder:
                self.recorder.write(g)
            time.sleep(period)
