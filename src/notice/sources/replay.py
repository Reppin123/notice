"""Record raw gaze to JSONL and replay it (time-shifted to now) for hardware-free testing."""
from __future__ import annotations

import json
import threading
import time

from ..core import AttentionState, GazePoint, IDTDetector


class GazeRecorder:
    def __init__(self, path: str):
        self.f = open(path, "a", buffering=1)

    def write(self, g: GazePoint):
        self.f.write(json.dumps({"t": g.t, "x": g.x, "y": g.y, "worn": g.worn}) + "\n")


class ReplayGaze:
    def __init__(self, state: AttentionState, path: str, speed: float = 1.0, loop: bool = False):
        self.state, self.path, self.speed, self.loop = state, path, speed, loop
        self.detector = IDTDetector(state.add_fixation)

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="replay-gaze").start()
        return self

    def _run(self):
        rows = [json.loads(l) for l in open(self.path) if l.strip()]
        while True:
            t0_rec, t0 = rows[0]["t"], time.time()
            for r in rows:
                due = t0 + (r["t"] - t0_rec) / self.speed
                if (d := due - time.time()) > 0:
                    time.sleep(d)
                g = GazePoint(time.time(), r["x"], r["y"], r.get("worn", True))
                self.state.add_gaze(g)
                self.detector.feed(g)
            if not self.loop:
                return
