"""Scene-camera frames on the Mac clock: a short ring of JPEGs, plus a webcam stand-in for dev."""
from __future__ import annotations

import bisect
import threading
import time
from collections import deque

import cv2
import numpy as np


class SceneBuffer:
    """Last ~10 s of scene frames as JPEG (a 1600x1200 BGR frame is 5.7 MB; a JPEG ~200 KB)."""

    def __init__(self, keep_s: float = 10.0):
        self.keep_s = keep_s
        self.frames: deque = deque()  # (t, jpeg bytes, (w, h))
        self.lock = threading.Lock()

    def add(self, t: float, bgr: np.ndarray):
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return
        with self.lock:
            self.frames.append((t, jpg.tobytes(), (bgr.shape[1], bgr.shape[0])))
            while self.frames and self.frames[0][0] < t - self.keep_s:
                self.frames.popleft()

    def nearest(self, t: float):
        with self.lock:
            if not self.frames:
                return None
            ts = [f[0] for f in self.frames]
            i = bisect.bisect_left(ts, t)
            cands = [self.frames[j] for j in (i - 1, i) if 0 <= j < len(self.frames)]
            return min(cands, key=lambda f: abs(f[0] - t))

    def latest(self):
        with self.lock:
            return self.frames[-1] if self.frames else None


def decode(jpg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def crop_around(bgr: np.ndarray, x: float, y: float, size: int = 400) -> np.ndarray:
    h, w = bgr.shape[:2]
    half = size // 2
    x0 = int(min(max(x - half, 0), max(w - size, 0)))
    y0 = int(min(max(y - half, 0), max(h - size, 0)))
    return bgr[y0:y0 + size, x0:x0 + size].copy()


def mark_gaze(bgr: np.ndarray, x: float, y: float, max_w: int = 1024) -> np.ndarray:
    """Full frame with a gaze circle, downscaled for the model (keeps tokens low)."""
    out = bgr.copy()
    r = max(18, out.shape[1] // 40)
    cv2.circle(out, (int(x), int(y)), r, (0, 0, 255), 4)
    cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
    if out.shape[1] > max_w:
        s = max_w / out.shape[1]
        out = cv2.resize(out, (max_w, int(out.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return out


def dhash(bgr: np.ndarray) -> int:
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def colour_hist(bgr: np.ndarray) -> np.ndarray:
    """Hue/saturation histogram: survives the object shifting or tilting in the crop, unlike dhash.
    On real Neon crops one held object scored 0.52-0.94 against itself and <= 0.26 against the room."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def same_colours(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


class WebcamScene:
    """Dev stand-in for Neon's scene camera. Mouse position (fraction of the screen) = gaze in the frame."""

    def __init__(self, buf: SceneBuffer, index: int = 0, fps: float = 10):
        self.buf, self.index, self.period = buf, index, 1.0 / fps
        self.size = None

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="webcam-scene").start()
        return self

    def _run(self):
        cap = cv2.VideoCapture(self.index)
        while True:
            ok, frame = cap.read()
            if ok:
                self.size = (frame.shape[1], frame.shape[0])
                self.buf.add(time.time(), frame)
            time.sleep(self.period)


class FileScene:
    """Dev stand-in: a still image (or looping video) as the scene camera. No people on camera."""

    def __init__(self, buf: SceneBuffer, path: str, fps: float = 10):
        self.buf, self.path, self.period = buf, path, 1.0 / fps
        self.size = None

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="file-scene").start()
        return self

    def _run(self):
        still = cv2.imread(self.path)
        cap = None if still is not None else cv2.VideoCapture(self.path)
        while True:
            if still is not None:
                frame = still
            else:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            self.size = (frame.shape[1], frame.shape[0])
            self.buf.add(time.time(), frame)
            time.sleep(self.period)
