"""Replay a Neon "Native Recording Data" folder as if the glasses were streaming now.

Gaze (200 Hz, scene px), on-device fixations and ~10 fps scene frames are re-timed to
Mac time.time() and pushed into the same AttentionState / SceneBuffer the live NeonGaze
feeds. Screen mapping is not done (x, y = None), as for world gaze. Fixations are added
when they END, as the live eye-events stream delivers them. Recordings made without the
fixation stream get fixations from IDTDetector on scene px instead.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pupil_labs.neon_recording as nr

from ..core import AttentionState, Fixation, GazePoint, IDTDetector

log = logging.getLogger("notice.neon_replay")


class NeonReplay:
    def __init__(self, state: AttentionState, scene, path: str, speed: float = 1.0, loop: bool = True,
                 scene_hz: float = 10.0):
        self.state, self.scene, self.path = state, scene, path
        self.speed, self.loop, self.scene_hz = speed, loop, scene_hz
        self.stats = {"gaze": 0, "fixations": 0, "frames": 0, "loops": 0}
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="neon-replay").start()
        return self

    def stop(self):
        self._stop.set()

    def _load(self, rec):
        g = rec.gaze
        gt = g.data["time"].astype(np.int64)
        gxy = np.column_stack([g.data["point_x"], g.data["point_y"]]).astype(float)
        try:
            w = rec.worn.data["worn"]
            worn = w > 0 if len(w) == len(gt) else np.ones(len(gt), bool)
        except nr.NeonRecording.SensorError:
            worn = np.ones(len(gt), bool)
        try:
            f = rec.fixations
            d = f.data
            order = np.argsort(d["stop_time"])
            fix = np.column_stack([d["start_time"], d["stop_time"]])[order].astype(np.int64)
            fxy = np.column_stack([d["mean_gaze_x"], d["mean_gaze_y"]]).astype(float)[order]
        except nr.NeonRecording.SensorError:
            log.info("no fixation stream in %s; detecting fixations from gaze", self.path)
            fix, fxy = None, None
        try:
            st = rec.scene.data["time"].astype(np.int64)
        except nr.NeonRecording.SensorError:
            st = np.zeros(0, np.int64)
        return gt, gxy, worn, fix, fxy, st

    def _run(self):
        rec = nr.open(self.path)
        gt, gxy, worn, fix, fxy, st = self._load(rec)
        rec0 = int(min(gt[0], st[0] if len(st) else gt[0]))
        stride_ns = int(1e9 * self.speed / self.scene_hz)  # recording time between replayed frames
        while not self._stop.is_set():
            t0 = time.time()
            to_mac = lambda ns: t0 + (ns - rec0) / 1e9 / self.speed  # noqa: E731
            detector = None
            if fix is None:
                def on_fix(f):
                    f.source = "neon"
                    self.state.add_fixation(f)
                    self.stats["fixations"] += 1
                detector = IDTDetector(on_fix)
            gi = fi = 0
            next_frame_ns = st[0] if len(st) else None
            while gi < len(gt) and not self._stop.is_set():
                now_ns = rec0 + int((time.time() - t0) * self.speed * 1e9)
                while gi < len(gt) and gt[gi] <= now_ns:
                    p = GazePoint(to_mac(gt[gi]), None, None, bool(worn[gi]), gxy[gi, 0], gxy[gi, 1])
                    self.state.add_gaze(p)
                    if detector:
                        detector.feed(p if p.worn else GazePoint(p.t, None, None, False))
                    self.stats["gaze"] += 1
                    gi += 1
                while fix is not None and fi < len(fix) and fix[fi, 1] <= now_ns:
                    self.state.add_fixation(Fixation(to_mac(fix[fi, 0]), to_mac(fix[fi, 1]), None, None,
                                                     float(fxy[fi, 0]), float(fxy[fi, 1]), source="neon"))
                    self.stats["fixations"] += 1
                    fi += 1
                if next_frame_ns is not None and next_frame_ns <= now_ns:
                    si = int(np.searchsorted(st, now_ns, side="right")) - 1  # newest due frame
                    self.scene.add(to_mac(st[si]), rec.scene[si].bgr)
                    self.stats["frames"] += 1
                    next_frame_ns = int(st[si]) + stride_ns if si + 1 < len(st) else None
                time.sleep(0.005)
            self.stats["loops"] += 1
            if not self.loop:
                break
        rec.close()
