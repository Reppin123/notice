"""Opt-in session recorder (--record-sessions): while capture is armed (on a call), save raw gaze,
fixations and scene frames (2 fps) so a run can be replayed for tuning or as a backup take.
Raw frames can include people, so this is OFF by default and pruned with the 24 h rule."""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from pathlib import Path

from .core import AttentionState, Fixation, GazePoint, IDTDetector
from .memory import DATA, KEEP_S

SESSIONS = DATA / "sessions"


class Recorder:
    def __init__(self, state: AttentionState, scene, memory, frame_period: float = 0.5):
        self.state, self.scene, self.memory, self.period = state, scene, memory, frame_period
        self.active = False
        self.dir: Path | None = None

    def prune(self):
        cutoff = time.time() - KEEP_S
        for d in SESSIONS.glob("*"):
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)

    async def run(self):
        SESSIONS.mkdir(parents=True, exist_ok=True)
        self.prune()
        last_g = last_fx_end = last_frame = 0.0
        gaze_f = fix_f = None
        while True:
            await asyncio.sleep(0.25)
            if not self.memory.armed:
                if self.active:
                    gaze_f.close(); fix_f.close()
                    self.active = False
                continue
            if not self.active:
                self.dir = SESSIONS / time.strftime("%Y%m%d-%H%M%S")
                (self.dir / "frames").mkdir(parents=True, exist_ok=True)
                gaze_f = open(self.dir / "gaze.jsonl", "a", buffering=1)
                fix_f = open(self.dir / "fixations.jsonl", "a", buffering=1)
                self.active = True
                last_g = last_fx_end = time.time()
            with self.state.lock:
                gs = [g for g in self.state.gaze if g.t > last_g]
                fs = [f for f in self.state.fixations if f.end > last_fx_end and f.end < time.time() - 0.5]
            for g in gs:
                gaze_f.write(json.dumps({"t": g.t, "x": g.x, "y": g.y, "sx": g.sx, "sy": g.sy, "worn": g.worn}) + "\n")
            if gs:
                last_g = gs[-1].t
            for f in fs:
                fix_f.write(json.dumps(f.to_dict()) + "\n")
            if fs:
                last_fx_end = max(f.end for f in fs)
            fr = self.scene.latest() if self.scene else None
            if fr and fr[0] - last_frame >= self.period:
                last_frame = fr[0]
                (self.dir / "frames" / f"{fr[0]:.3f}.jpg").write_bytes(fr[1])


class SessionReplay:
    """--gaze session:<dir> — replay a recorded session (time-shifted to now), frames included."""

    def __init__(self, state: AttentionState, scene, path: str, loop: bool = True):
        self.state, self.scene, self.dir, self.loop = state, scene, Path(path), loop

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="session-replay").start()
        return self

    def _run(self):
        import cv2
        gaze = [json.loads(l) for l in open(self.dir / "gaze.jsonl") if l.strip()]
        fixes = [json.loads(l) for l in open(self.dir / "fixations.jsonl") if l.strip()]
        frames = sorted(self.dir.glob("frames/*.jpg"), key=lambda p: float(p.stem))
        events = ([(r["t"], "g", r) for r in gaze] + [(r["end"], "f", r) for r in fixes]
                  + [(float(p.stem), "v", p) for p in frames])
        events.sort(key=lambda e: e[0])
        while events:
            t0_rec, t0 = events[0][0], time.time()
            for t, kind, r in events:
                due = t0 + (t - t0_rec)
                if (d := due - time.time()) > 0:
                    time.sleep(d)
                shift = t0 - t0_rec
                if kind == "g":
                    self.state.add_gaze(GazePoint(r["t"] + shift, r["x"], r["y"], r.get("worn", True), r.get("sx"), r.get("sy")))
                elif kind == "f":
                    self.state.add_fixation(Fixation(r["start"] + shift, r["end"] + shift, r["x"], r["y"],
                                                     r.get("scene_x"), r.get("scene_y"), "replay"))
                else:
                    img = cv2.imread(str(r))
                    if img is not None:
                        self.scene.add(t + shift, img)
            if not self.loop:
                return
