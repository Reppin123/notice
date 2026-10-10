"""Shared attention state: fixations on one clock (Mac unix seconds)."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field


@dataclass
class Fixation:
    start: float  # Mac unix seconds
    end: float
    x: float | None  # screen points (global, top-left origin); None when off-screen
    y: float | None
    scene_x: float | None = None  # Neon scene-camera pixels (world mode)
    scene_y: float | None = None
    source: str = "sim"

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def on_screen(self) -> bool:
        return self.x is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_ms"] = round(self.duration * 1000)
        return d


@dataclass
class GazePoint:
    t: float
    x: float | None
    y: float | None
    worn: bool = True
    sx: float | None = None  # scene-camera px (world)
    sy: float | None = None


@dataclass
class AttentionState:
    """Ring buffers of recent gaze + fixations. Thread-safe; sources write, readers query."""

    keep_s: float = 60.0
    fixations: deque = field(default_factory=lambda: deque(maxlen=2000))
    gaze: deque = field(default_factory=lambda: deque(maxlen=2000))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_fixation(self, f: Fixation) -> None:
        with self.lock:
            self.fixations.append(f)
            cutoff = f.end - self.keep_s
            while self.fixations and self.fixations[0].end < cutoff:
                self.fixations.popleft()

    def add_gaze(self, g: GazePoint) -> None:
        with self.lock:
            self.gaze.append(g)

    def fixations_between(self, t0: float, t1: float) -> list[Fixation]:
        with self.lock:
            return [f for f in self.fixations if f.end >= t0 and f.start <= t1]

    def last_gaze(self) -> GazePoint | None:
        with self.lock:
            return self.gaze[-1] if self.gaze else None

    def latest_fixation(self) -> Fixation | None:
        with self.lock:
            return self.fixations[-1] if self.fixations else None

    def on_screen_between(self, t0: float, t1: float) -> float | None:
        """Fraction of gaze samples on screen in [t0, t1]; None if no samples."""
        with self.lock:
            pts = [g for g in self.gaze if t0 <= g.t <= t1]
        if not pts:
            return None
        return sum(1 for g in pts if g.x is not None and g.worn) / len(pts)

    def on_screen_fraction(self, window_s: float = 0.6, now: float | None = None) -> float:
        now = now or time.time()
        with self.lock:
            pts = [g for g in self.gaze if g.t >= now - window_s]
        if not pts:
            return 0.0
        return sum(1 for g in pts if g.x is not None and g.worn) / len(pts)


class IDTDetector:
    """Dispersion-threshold fixation detector for raw gaze (simulator / replay).

    Neon emits fixations on-device; this gives the simulator the same shape of data. An
    in-progress fixation is published as soon as it passes min_duration and then grows in
    place, so "this" said while still looking at something resolves immediately.
    """

    def __init__(self, on_fixation, max_dispersion: float = 45.0, min_duration: float = 0.1):
        self.on_fixation = on_fixation
        self.max_dispersion = max_dispersion
        self.min_duration = min_duration
        self.window: list[GazePoint] = []
        self.current: Fixation | None = None
        self.scene_of = None  # optional fn(GazePoint) -> (sx, sy) for world-mode sims

    @staticmethod
    def _xy(p):
        return (p.x, p.y) if p.x is not None else (p.sx, p.sy)

    def _dispersion(self, pts) -> float:
        xs = [self._xy(p)[0] for p in pts]
        ys = [self._xy(p)[1] for p in pts]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    def _update(self):
        w = self.window
        if not w or w[-1].t - w[0].t < self.min_duration:
            return
        n = len(w)
        x = y = None
        if w[-1].x is not None:
            x, y = sum(p.x for p in w) / n, sum(p.y for p in w) / n
        sx = sy = None
        if w[-1].sx is not None:
            sx, sy = sum(p.sx for p in w) / n, sum(p.sy for p in w) / n
        if self.current is None:
            self.current = Fixation(w[0].t, w[-1].t, x, y, sx, sy)
            self.on_fixation(self.current)
        else:
            c = self.current
            c.end, c.x, c.y, c.scene_x, c.scene_y = w[-1].t, x, y, sx, sy

    def _flush(self):
        self._update()
        self.window, self.current = [], None

    def feed(self, g: GazePoint) -> None:
        if g.x is None and g.sx is None:
            self._flush()
            return
        self.window.append(g)
        if len(self.window) > 1 and self._dispersion(self.window) > self.max_dispersion:
            last = self.window.pop()
            self._flush()
            self.window = [last]
        else:
            self._update()
